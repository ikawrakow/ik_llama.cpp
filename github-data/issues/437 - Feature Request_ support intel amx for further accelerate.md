### ✨ [#437](https://github.com/ikawrakow/ik_llama.cpp/issues/437) - Feature Request: support intel amx for further accelerate

| **Author** | `zhaoyukoon` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-05-20 |
| **Updated** | 2025-07-06 |

---

#### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

I learned from [ktransformers-Intel-AMX](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md) that amx instruction can further improve inference speed for MoE models. 

Is there any plan to support amx in ik_llama? Thanks!

### Motivation

Ktransformer kernel can achieve 21 TFLOPS of BF16 throughput and 35 TOPS of Int8 throughput on Xeon4 CPUs — about 4× faster than PyTorch’s general AMX kernel. For DeepSeek-V3, pairing a Xeon4 CPU with a single RTX 4090 GPU achieves 418 tokens/s end-to-end throughput, close to the performance of multi-machine, multi-GPU setups. KTransformers’ AMX kernel is the first AMX kernel specifically designed for MoE inference scenarios, significantly lowering the hardware barrier for large model deployment and enabling more developers to enjoy GPU cluster level inference experiences at lower cost.

### Possible Implementation

_No response_

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-20** at **09:13:24**:<br>

If someone gives me access to a system with AMX support, then sure, I would work on that.

But out of curiosity, do you have a performance comparison between ik_llama.cpp and KTransformers on the same system?

---

👤 **zhaoyukoon** commented the **2025-05-20** at **10:11:14**:<br>

> If someone gives me access to a system with AMX support, then sure, I would work on that.
> 
> But out of curiosity, do you have a performance comparison between ik_llama.cpp and KTransformers on the same system?

I can access a server equipped with AMX Intel CPUs, however I have no permission to add other uses. I can help to run test on this server.

I tested ktransformers on another AMD server with 24GB 4090D, which can get 15+ tokens/s decoding speed. I have not tested ik_llama yet, I learned that llama.cpp can get 7 tokens/s on pure CPU.

https://github.com/XuanwuLab/llama.cpp_deepseek/blob/main/llama-mmap.cpp

https://mp.weixin.qq.com/s/vIrvbVJ6Nv00Ehre1zZwMw [In Chinese]

---

👤 **ikawrakow** commented the **2025-05-20** at **10:37:39**:<br>

I cannot say that I'm particularly impressed with the performance reported in [ktransformers-Intel-AMX](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md). For convenience here is what they report:

![Image](https://github.com/user-attachments/assets/cba0d8b5-4685-474f-9882-a2418902dc0a)

My system is Ryzen-7950X CPU + 4080 GPU. Based on benchmarks from [here](https://www.cpubenchmark.net/cpu.php?id=5684) and [here](https://www.cpubenchmark.net/cpu.php?id=5031&cpu=AMD+Ryzen+9+7950X), my CPU is only marginally faster than their "consumer" level system with Intel-14900KF + 4090 GPU. I don't have enough RAM to run Qwen3-235-A22B, but here is what I get for Qwen3-30B-A3B quantized with `IQ4_XS` (so corresponds to their 4-bit result) with `ik_llama.cpp`:

### CPU only

| model                |       size | backend    |          test |              t/s |   
| -------------------- | ---------: | ---------- | ------------: | ---------------: |
| qwen3moe 30B IQ4_XS   |  15.24 GiB | CPU        |         pp512 |    480.78 ± 2.11 |  
| qwen3moe 30B IQ4_XS   |  15.24 GiB | CPU        |         tg128 |     29.17 ± 0.08 

Here `pp512` corresponds to what they call "prefill" and `tg128` is what they call "decode". So, even without a GPU `ik_llama.cpp` beets their prefill performance by 2X, and is faster than their "4-way decode" performance on the "consumer" level system that has roughly the same speed as mine.

### CPU+GPU

Here speed depends on how many layers I offload to the GPU. But let's keep 18 layers on the CPU so I have enough VRAM for the maximum context of 41,000 tokens on my paltry 16 GB GPU. Here is what I get with that:

| model                |       size | backend    |          test |              t/s |    
| -------------------- | ---------: | ---------- | ------------: | ---------------: |
| qwen3moe 30B IQ4_XS  |  15.24 GiB | CUDA       |        pp2048 |  3039.84 ± 24.96 |  
| qwen3moe 30B IQ4_XS  |  15.24 GiB | CUDA       |         tg128 |     77.44 ± 0.39 |    

So, 15X their prefill performance and 3X their "4-way decode" performance ("consumer level" system), and 8.7X prefill, 1.5X "4-way decode" (Xeon 4 workstation).

---

👤 **ikawrakow** commented the **2025-05-20** at **10:48:25**:<br>

> I can access a server equipped with AMX Intel CPUs, however I have no permission to add other uses. I can help to run test on this server.

This will be way too tedious. I have to build with AMX instructions enabled, then you test and find gibberish, then I second-guess where is the bug, change something, you test and find gibberish, rinse and repeat. I have to have access to the AMX-enabled system while writing the code.

---

👤 **zhaoyukoon** commented the **2025-05-20** at **11:08:03**:<br>

> > I can access a server equipped with AMX Intel CPUs, however I have no permission to add other uses. I can help to run test on this server.
> 
> This will be way too tedious. I have to build with AMX instructions enabled, then you test and find gibberish, then I second-guess where is the bug, change something, you test and find gibberish, rinse and repeat. I have to have access to the AMX-enabled system while writing the code.

Do you have any requirements on CPU and memory for development? Is server with 16 vCPU AMX and 32GB enough?

---

👤 **ikawrakow** commented the **2025-05-20** at **11:47:29**:<br>

> Do you have any requirements on CPU and memory for development? Is server with 16 vCPU AMX and 32GB enough?

Yes, that's should be enough for development.

But before you go and rent a cloud instance, let's start by you fist testing `ik_llama.cpp` on your system and comparing performance to KTransformers.

Let's also make sure that the expectations are aligned:
* It is extremely unlikely AMX will improve token generation (TG) speed
* It is very unlikely AMX will improve prefill speed for hybrid CPU/GPU inference for most models. Only the LLaMA-4 models may get faster
* AMX will improve prefill performance for **CPU-only** inference compared to vanilla `AVX2` implementations such as what you have in `llama.cpp` or KTransformers. If it will improve performance compared to the existing `ik_llama.cpp` implementation remains to be seen.

---

👤 **kirnat** commented the **2025-05-20** at **14:17:01**:<br>

While I’d be excited to see AMX support, I can’t say the kTransformers Qwen3 benchmark proves its usefulness. I can’t verify the pp/tg window sizes or the exact model they used, but as an inexact comparison, I got the below results in ik_llama for Qwen3 235B with Xeon 8480 (ES), 8-channel 4800MT DDR5 and a blackwell GPU.

Model used:
**unsloth/Qwen3-235B-A22B-GGUF/UD-Q4_K_XL/Qwen3-235B-A22B-UD-Q4_K_XL**
|       size |     params | backend    | ngl | threads | n_batch | n_ubatch | fa | rtr | fmoe |          test |              t/s |
| ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -: | --: | ---: | ------------: | ---------------: |
| 124.91 GiB |   235.09 B | CUDA       |  93 |      52 |    8192 |     8192 |  1 |   1 |    1 |        pp2048 |    192.02 ± 0.06 |
| 124.91 GiB |   235.09 B | CUDA       |  93 |      52 |    8192 |     8192 |  1 |   1 |    1 |       pp16384 |    185.33 ± 0.34 |
| 124.91 GiB |   235.09 B | CUDA       |  93 |      52 |    8192 |     8192 |  1 |   1 |    1 |         tg512 |     18.74 ± 0.02 |
| 124.91 GiB |   235.09 B | CUDA       |  93 |      52 |    8192 |     8192 |  1 |   1 |    1 |        tg2048 |     18.58 ± 0.03 |

The 30B model performs really well on CPU only, below is with GPU hidden.

Model used:
**unsloth/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-UD-Q4_K_XL**
|       size |     params | backend    | ngl | threads | fa | fmoe |          test |              t/s |
| ---------: | ---------: | ---------- | --: | ------: | -: | ---: | ------------: | ---------------: |
|  16.49 GiB |    30.53 B | CUDA       |   0 |      32 |  1 |    1 |         pp512 |    510.65 ± 2.49 |
|  16.49 GiB |    30.53 B | CUDA       |   0 |      32 |  1 |    1 |        pp2048 |    454.62 ± 0.18 |
|  16.49 GiB |    30.53 B | CUDA       |   0 |      32 |  1 |    1 |         tg128 |     69.77 ± 0.02 |
|  16.49 GiB |    30.53 B | CUDA       |   0 |      32 |  1 |    1 |         tg512 |     69.15 ± 0.01 |

Thanks a lot for the impressive work ikawrakow!

---

👤 **kirnat** commented the **2025-05-20** at **14:17:01**:<br>

While I’d be excited to see AMX support, I can’t say the kTransformers Qwen3 benchmark proves its usefulness. I can’t verify the pp/tg window sizes or the exact model they used, but as an inexact comparison, I got the below results for Qwen3 235B with Xeon 8480 (ES), 8-channel 4800MT DDR5 and a blackwell GPU.

Model used:
**unsloth/Qwen3-235B-A22B-GGUF/UD-Q4_K_XL/Qwen3-235B-A22B-UD-Q4_K_XL**
|       size |     params | backend    | ngl | threads | n_batch | n_ubatch | fa | rtr | fmoe |          test |              t/s |
| ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -: | --: | ---: | ------------: | ---------------: |
| 124.91 GiB |   235.09 B | CUDA       |  93 |      52 |    8192 |     8192 |  1 |   1 |    1 |        pp2048 |    192.02 ± 0.06 |
| 124.91 GiB |   235.09 B | CUDA       |  93 |      52 |    8192 |     8192 |  1 |   1 |    1 |       pp16384 |    185.33 ± 0.34 |
| 124.91 GiB |   235.09 B | CUDA       |  93 |      52 |    8192 |     8192 |  1 |   1 |    1 |         tg512 |     18.74 ± 0.02 |
| 124.91 GiB |   235.09 B | CUDA       |  93 |      52 |    8192 |     8192 |  1 |   1 |    1 |        tg2048 |     18.58 ± 0.03 |

The 30B model performs really well on CPU only, below is with GPU hidden.

Model used:
**unsloth/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-UD-Q4_K_XL**
|       size |     params | backend    | ngl | threads | fa | fmoe |          test |              t/s |
| ---------: | ---------: | ---------- | --: | ------: | -: | ---: | ------------: | ---------------: |
|  16.49 GiB |    30.53 B | CUDA       |   0 |      32 |  1 |    1 |         pp512 |    510.65 ± 2.49 |
|  16.49 GiB |    30.53 B | CUDA       |   0 |      32 |  1 |    1 |        pp2048 |    454.62 ± 0.18 |
|  16.49 GiB |    30.53 B | CUDA       |   0 |      32 |  1 |    1 |         tg128 |     69.77 ± 0.02 |
|  16.49 GiB |    30.53 B | CUDA       |   0 |      32 |  1 |    1 |         tg512 |     69.15 ± 0.01 |

Thanks a lot for the impressive work ikawrakow!

---

👤 **ikawrakow** commented the **2025-05-20** at **15:06:56**:<br>

Has anyone tried mainline `llama.cpp` AMX implementation?

---

👤 **zhaoyukoon** commented the **2025-05-20** at **16:09:15**:<br>

> Has anyone tried mainline `llama.cpp` AMX implementation?

https://github.com/ggml-org/llama.cpp/issues/12003

It seems that llama.cpp supports amx

---

👤 **ikawrakow** commented the **2025-05-20** at **16:28:01**:<br>

> It seems that llama.cpp supports amx.

That's why I asked if somebody has tried. It would be even more interesting if someone has compared `llama.cpp` performance to `ik_llama.cpp` on an AMX CPU.

---

👤 **kirnat** commented the **2025-05-20** at **19:18:44**:<br>

### Confirming AMX buffer
**llama.cpp/build/bin/llama-cli -m ./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf**
load_tensors: loading model tensors, this can take a while... (mmap = true)
load_tensors:   CPU_Mapped model buffer size =  4685.30 MiB
load_tensors:          AMX model buffer size =  4491.48 MiB
........................................................................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 4096
llama_context: n_ctx_per_seq = 4096
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = 0
llama_context: freq_base     = 500000.0
llama_context: freq_scale    = 1
llama_context: n_ctx_per_seq (4096) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
llama_context:        CPU  output buffer size =     0.49 MiB
llama_kv_cache_unified: kv_size = 4096, type_k = 'f16', type_v = 'f16', n_layer = 32, can_shift = 1, padding = 32
llama_kv_cache_unified:        CPU KV buffer size =   512.00 MiB
llama_kv_cache_unified: KV self size  =  512.00 MiB, K (f16):  256.00 MiB, V (f16):  256.00 MiB
llama_context:        CPU compute buffer size =   296.01 MiB


### llama.cpp bench
llama.cpp/build/bin/llama-bench -m ./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -t 52 -fa 1
| model                          |       size |     params | backend    | threads | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -: | --------------: | -------------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      52 |  1 |           pp512 |        228.18 ± 0.03 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CPU        |      52 |  1 |           tg128 |         37.28 ± 0.01 |

build: e3a9421b (5389)


### ik_llama bench
ik_llama.cpp/build/bin/llama-bench -ngl 0 -m ./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -t 52 -fa 1
ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected
| model                          |       size |     params | backend    | ngl | threads | fa |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -: | ------------: | ---------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CUDA       |   0 |      52 |  1 |         pp512 |    348.00 ± 0.43 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CUDA       |   0 |      52 |  1 |         tg128 |     42.48 ± 0.03 |

build: 2ec2229f (3702)

---

Let me know if you want me to test with another model specific settings. I used high thread count since it helps prompt processing while penalizes token generation slightly, but not too much in this case.

---

👤 **ikawrakow** commented the **2025-05-20** at **19:33:34**:<br>

Thanks!

You could try adding `-rtr 1` to the `ik_llama.cpp` benchmark run. This normally gives a significant boost in PP performance.

---

👤 **kirnat** commented the **2025-05-20** at **20:47:09**:<br>

I hadn't even considered it for CPU only inference. I have used it alot day to day for hybrid inference with great results.

Same settings as above and GPU hidden but with rtr enabled.

| model                          |       size |     params | backend    | ngl | threads | fa | rtr |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -: | --: | ------------: | ---------------: |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CUDA       |   0 |      52 |  1 |   1 |         pp512 |    444.89 ± 0.96 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CUDA       |   0 |      52 |  1 |   1 |       pp16384 |    267.07 ± 3.60 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CUDA       |   0 |      52 |  1 |   1 |         tg128 |     43.21 ± 0.03 |
| llama 8B Q4_K - Medium         |   4.58 GiB |     8.03 B | CUDA       |   0 |      52 |  1 |   1 |        tg2048 |     41.39 ± 0.01 |

Still amazed how relatively low the slow down is in ik at larger context sizes. This tranlates to Qwen3, DeepSeek v3 and Llama 4 Maverick as well.

---

👤 **ikawrakow** commented the **2025-05-21** at **04:39:40**:<br>

So, `ik_llama.cpp` without AMX is nearly two times faster than `llama.cpp` with AMX.

---

👤 **mtcl** commented the **2025-06-08** at **06:03:35**:<br>

Specifically for the new r1-0528 (but results are similar for v3-0324):

I have an amx supported pc, and I can confirm that performance for ktransformers is noticibly better than ik_llama and llama.cpp (in that same order) for prompt processing. In general I get about 50 prefill (prompt processing) on ktransformers and 10 tk/s on generation. It is the prompt processing that has massive benefits on ktransformers. I get less than half on the prompt processing on ik_llama.cpp. token generation is comparable (but ktransformers has about 10% advantage). 

With KTransformers, I can only fit 24K context length on single 4090 on my PC (512 GB DDR5 ram, 8 channel, 4800 MHz), whereas I can fit 32K context length with similar quants on ik_llama

Another difference with KTransformers is that I can make a hybrid model q4km_fp8 hybrid and all of fp8 processing is done on the GPU. Apparantly they have some special kernal that helps it speed it up on the GPU with FP8 processing.

I have been following @ubergarm 's quants and guide to run it on ik_llama.

I love your work here @ikawrakow and I would love to contribute in anyway to make this project better than ktransformers! If you need me to run anything please let me know!

---

👤 **ikawrakow** commented the **2025-06-08** at **06:26:52**:<br>

> I have an amx supported pc, and I can confirm that performance for ktransformers is noticibly better than ik_llama and llama.cpp (in that same order) for prompt processing. In general I get about 50 prefill (prompt processing) on ktransformers and 10 tk/s on generation. It is the prompt processing that has massive benefits on ktransformers. I get less than half on the prompt processing on ik_llama.cpp. token generation is comparable (but ktransformers has about 10% advantage).

If you share your `ik_llama.cpp` command line you used to measure performance, perhaps we can help you make it faster.  You didn't share the specs of your CPU and GPU, but 25 t/s prefill given 9 t/s generation does not sound reasonable for `ik_llama.cpp`.

---

👤 **ubergarm** commented the **2025-06-08** at **15:48:30**:<br>

@mtcl 

Hey thanks again for your youtube video showing your OpenAI compatible wrapper working with ik_llama.cpp and one of my quants! Very cool!

>  It is the prompt processing that has massive benefits on ktransformers.

ik has shown me and others have reported success increasing prompt processing (prefill on ktransformers) by increasing batch size e.g. `-b 4096 -ub 4096` assuming you free up enough VRAM using `-ctk q8_0` or lower context a bit etc. You might have to play with exact numbers to adjust speed / VRAM usage tradeoff and might not work on all setups. I can hit over 200 tok/sec prompt processing with some of my R1-0528 quants using this.

> Another difference with KTransformers is that I can make a hybrid model q4km_fp8 hybrid and all of fp8 processing is done on the GPU. 

They do have some goofy hybrid quants that use fp8 for GPU offload layers which requires CUDA 40 series or newer that support fp8 E4M3. But the 3090 and older only supports fp8 E5M2 so those ktransformers kernels are not widely applicable. My quants use high quality iq4_ks, iq5_ks, or full q8_0 for those tensors which will likely be better quality and more performant across a wider variety of systems.

Finally, last I checked, ktransformers performance *tanked* when attempting to offload additional layers onto GPU given how they were relying on cuda graphs. So after fiddling with those confusing yaml files performance was *worse* when using *more* VRAM... This was a couple months ago so ymmv.  So multi-GPU story on ik seems much better imo unless things changed radically over on ktransformers.

But as ik says, share your commands and might be able to get you a boost.

---

👤 **mtcl** commented the **2025-06-09** at **00:26:07**:<br>

@ubergarm and @ikawrakow  Below is for Qwen3-235 billion parameter model. Tthank you for the pointers! For the qwen models, i added "-b 2048 -ub 2048" and that resulted in the max speeds for me. I am getting 150+ prompt processing tk/seconds on that now! That is insane! 

This was my original command
```bash
CUDA_VISIBLE_DEVICES="1" ./build/bin/llama-server \
  --model /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf \
  --alias ubergarm/Qwen3-235B-A22B-mix-IQ3_K \
  -fa \
  -ctk q4_0 -ctv q4_0 \
  -c 32768 \
  -fmoe \
  -amb 512 \
  -rtr \
  -ot blk\.1[2-9]\.ffn.*=CPU \
  -ot blk\.[2-8][0-9]\.ffn.*=CPU \
  -ot blk\.9[0-3]\.ffn.*=CPU \
  -ngl 99 \
  --threads 57 \
  --host 0.0.0.0 \
  --port 10002
```
I was getting very low prompt processing with this, under 50. After your recommendation, I switched the command to this:

```bash
CUDA_VISIBLE_DEVICES="1" ./build/bin/llama-server \
  --model /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf \
  --alias ubergarm/Qwen3-235B-A22B-mix-IQ3_K \
  -fa \
  -ctk q4_0 -ctv q4_0 \
  -c 32768 \
  -fmoe \
  -b 2048 -ub 2048 \
  -amb 512 \
  -rtr \
  -ot blk\.1[2-9]\.ffn.*=CPU \
  -ot blk\.[2-8][0-9]\.ffn.*=CPU \
  -ot blk\.9[0-3]\.ffn.*=CPU \
  -ngl 99 \
  --threads 57 \
  --host 0.0.0.0 \
  --port 10002
```

DeepSeek-R1-0528
You see how I have "ctk q4_0" in there right? It works for the Qwen model but not for the DeepSeek R1 model. 

This is my command before:

```bash
CUDA_VISIBLE_DEVICES="1" ./build/bin/llama-server \
    --model /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf \
    --alias ubergarm/DeepSeek-R1-0528-GGUF \
    --ctx-size 32768 \
    -ctk q8_0 \
    -mla 3 -fa \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```

This is my command after:

```bash
CUDA_VISIBLE_DEVICES="1" ./build/bin/llama-server \
    --model /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf \
    --alias ubergarm/DeepSeek-R1-0528-GGUF \
    --ctx-size 32768 \
    -ctk q4_0 \
    -mla 3 -fa \
    -b 2048 -ub 2048 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```

if i try to switch the ctk from Q8 to Q4 it crashes with below error:

```
INFO [   launch_slot_with_task] slot is processing task | tid="135217032957952" timestamp=1749426210 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="135217032957952" timestamp=1749426210 id_slot=0 id_task=0 p0=0
ggml_cuda_cpy_fn: unsupported type combination (q4_0 to f16)
ggml_cuda_cpy_fn: 64 x 2048 x 1; 324 x 10616832 10616832 -> 64 x 2048 x 1; 128 x 262144 x 262144
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/ggml-cuda/cpy.cu:718: fatal error



Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
Aborted (core dumped)
```

But if i keep the ctk as q8_0, i can have the context size of 24K with about 45 prompt processing speed, which is comparable to KTransformers. 

```bash
CUDA_VISIBLE_DEVICES="1" ./build/bin/llama-server \
    --model /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf \
    --alias ubergarm/DeepSeek-R1-0528-GGUF \
    --ctx-size 24576 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 2048 -ub 2048 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```
I will make a full video on this and will post this, unedited version of it, so that you can see everything in the process.

---

👤 **mtcl** commented the **2025-06-09** at **00:27:01**:<br>

@ubergarm Would you be able to post a guide on how to make the IQ4 version of the Qwen Model?

---

👤 **ikawrakow** commented the **2025-06-09** at **04:18:48**:<br>

@mtcl 

What is the model you are running with KTransformers? 

On the "crash": the DeepSeek self attention mechanism is special (different from basically any other model out there), so only `f16` and `Q8_0` can be used for KV cache. But even if it was supported, I would never use `Q4_0` for KV cache as the quality degradation is jsut too much for my taste. The lowest I would go (and only if desperate to reduce VRAM usage) would be `Q6_0` (but that is not supported for DeepSeek models).

---

👤 **mtcl** commented the **2025-06-09** at **04:24:19**:<br>

> [@mtcl](https://github.com/mtcl)
> 
> What is the model you are running with KTransformers?
> 
> On the "crash": the DeepSeek self attention mechanism is special (different from basically any other model out there), so only `f16` and `Q8_0` can be used for KV cache. But even if it was supported, I would never use `Q4_0` for KV cache as the quality degradation is jsut too much for my taste. The lowest I would go (and only if desperate to reduce VRAM usage) would be `Q6_0` (but that is not supported for DeepSeek models).

I am running @ubergarm 's IQ3_K_R4 model located here:

https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/tree/main/IQ3_K_R4

---

👤 **ikawrakow** commented the **2025-06-09** at **04:25:44**:<br>

That is with `ik_llama.cpp`. My question was what model are you running with KTransformers?

---

👤 **mtcl** commented the **2025-06-09** at **04:30:16**:<br>

> That is with `ik_llama.cpp`. My question was what model are you running with KTransformers?

oh sorry! I understand now, i am running a Q4_K_M-FP8 hybrid model, if you want to see how i create the model here is the video walkthrough of it: https://www.youtube.com/watch?v=Xui3_bA26LE 
Essentially ktransformers team provides a merge script to create these hybrid models out there. 

do you know if by using multiple 4090s i can increase context limit? I am also getting a 5090 tomorrow, so potentially it will help with more context on one GPU.

---

👤 **ikawrakow** commented the **2025-06-09** at **04:41:50**:<br>

> do you know if by using multiple 4090s i cna increate context limit? I am also getting a 5090 tomorrow, so potentially it will help with more context on one GPU.

Yes, some people with multiple GPU's have reported running full context length. Also, when you have more than 24 GB VRAM you can use `-b 4096 -ub 4096` and that will give another factor of nearly 2 increase in prefill performance. Some people have reported ever 200 t/s prefill with DeepSeek-R1/V3. @ubergarm has reported 100+ t/s running CPU-only. I don't have the hardware to run the DeepSeek models, but If I had enough RAM in my Ryzen-7950X box, I expect to get in the range of 50 t/s CPU-only using just this <$500 CPU (I hit 700 t/s with the 16B parameter DeepSeek-Lite that has 15X fewer active parameters than R1/V3).

---

👤 **mtcl** commented the **2025-06-09** at **04:45:10**:<br>

Can you please help me in modifying this command to get more context length with 2X4090 setup. 

```bash
CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-server \
    --model /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf \
    --alias ubergarm/DeepSeek-R1-0528-GGUF \
    --ctx-size 32768 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 2048 -ub 2048 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    -ot "blk\.(3|4)\.ffn_.*=CUDA0" \
    -ot "blk\.(5|6)\.ffn_.*=CUDA1" \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```

---

👤 **ikawrakow** commented the **2025-06-09** at **04:48:05**:<br>

Can you post the log? I don't know by heart how much VRAM gets used for model weights and KV cache, and how big CUDA compute buffers are.

---

👤 **ikawrakow** commented the **2025-06-09** at **05:06:31**:<br>

I think if you are able to offload two layers of experts per GPU you have in the range of 11 GB free on each GPU excuding the experts. It is likely that if you don't offload any experts to the GPU, you can a) nearly double prefill speed by using `-b 4096 -ub 4096` or b) increase context length to at least 65k tokens, or c) do a) and b).

---

👤 **mtcl** commented the **2025-06-09** at **05:32:06**:<br>

ok I posted the whole video here, showing every command i ran with all the log outputs. 

https://www.youtube.com/watch?v=kDhu0siTvEg



> I think if you are able to offload two layers of experts per GPU you have in the range of 11 GB free on each GPU excuding the experts. It is likely that if you don't offload any experts to the GPU, you can a) nearly double prefill speed by using `-b 4096 -ub 4096` or b) increase context length to at least 65k tokens, or c) do a) and b).

I am trying to understand how do i achieve this. What command can i run to give you the log here, can you please let me know?

---

👤 **mtcl** commented the **2025-06-09** at **05:46:15**:<br>

i tried modifying the command like this, but i get error:

(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ 
```bash
CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-server \
    --model /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf \
    --alias ubergarm/DeepSeek-R1-0528-GGUF \
    --ctx-size 32768 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 2048 -ub 2048 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    -ot "blk\.(3)\.ffn_.*=CUDA0" \
    -ot "blk\.(5)\.ffn_.*=CUDA1" \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```

```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
INFO [                    main] build info | tid="134935116726272" timestamp=1749447820 build=3737 commit="58f08e43"
INFO [                    main] system info | tid="134935116726272" timestamp=1749447820 n_threads=57 n_threads_batch=-1 total_threads=112 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 6 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 52 key-value pairs and 1147 tensors from /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 0528
llama_model_loader: - kv   3:                            general.version str              = 0528
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek-R1
llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
llama_model_loader: - kv   6:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   7:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   8:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   9:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  10:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  11:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  12:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  13: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  15:                          general.file_type u32              = 339
llama_model_loader: - kv  16:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  17:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  18:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  19:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  20:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  21:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  22:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  23:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  24:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  25:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  26:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  27:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  28:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  29:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  30:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  31: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  32: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  33:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  34:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  36:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  37:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  38:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  39:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  40:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  41:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  42:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  43:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  44:               general.quantization_version u32              = 2
llama_model_loader: - kv  45:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/DeepSeek-R1...
llama_model_loader: - kv  46:                   quantize.imatrix.dataset str              = ubergarm-imatrix-calibration-corpus-v...
llama_model_loader: - kv  47:             quantize.imatrix.entries_count i32              = 721
llama_model_loader: - kv  48:              quantize.imatrix.chunks_count i32              = 812
llama_model_loader: - kv  49:                                   split.no u16              = 0
llama_model_loader: - kv  50:                                split.count u16              = 7
llama_model_loader: - kv  51:                        split.tensors.count i32              = 1147
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq3_k_r4:  116 tensors
llama_model_loader: - type iq4_ks_r4:   58 tensors
llm_load_vocab: special tokens cache size = 818
llm_load_vocab: token to piece cache size = 0.8223 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 129280
llm_load_print_meta: n_merges         = 127741
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 163840
llm_load_print_meta: n_embd           = 7168
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 128
llm_load_print_meta: n_head_kv        = 128
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 24576
llm_load_print_meta: n_embd_v_gqa     = 16384
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 18432
llm_load_print_meta: n_expert         = 256
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 0.025
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = IQ3_K_R4 - 3.4325 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 300.938 GiB (3.847 BPW) 
llm_load_print_meta: repeating layers = 299.104 GiB (3.834 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1 0528
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: LF token         = 131 'Ä'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_layer_dense_lead   = 3
llm_load_print_meta: n_lora_q             = 1536
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 2048
llm_load_print_meta: n_expert_shared      = 1
llm_load_print_meta: expert_weights_scale = 2.5
llm_load_print_meta: expert_weights_norm  = 1
llm_load_print_meta: expert_gating_func   = sigmoid
llm_load_print_meta: rope_yarn_log_mul    = 0.1000
llm_load_tensors: ggml ctx size =    1.40 MiB
Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 36486.67 MiB
llm_load_tensors:        CPU buffer size = 43905.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 43905.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 44473.21 MiB
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 13995.99 MiB
llm_load_tensors:      CUDA1 buffer size = 13730.03 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 2048
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =   592.89 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   573.76 MiB
llama_new_context_with_model: KV self size  = 1166.62 MiB, c^KV (q8_0): 1166.62 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=4)
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1130415.93 MiB on device 0: cudaMalloc failed: out of memory
ggml_gallocr_reserve_n: failed to allocate CUDA0 buffer of size 1185327012864
llama_new_context_with_model: failed to allocate compute buffers
llama_init_from_gpt_params: error: failed to create context with model '/media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf'
 ERR [              load_model] unable to load model | tid="134935116726272" timestamp=1749447880 model="/media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf"
Segmentation fault (core dumped)
```

---

👤 **ikawrakow** commented the **2025-06-09** at **05:49:54**:<br>

Try `cmake  -DGGML_SCHED_MAX_COPIES=1 ...`

---

👤 **mtcl** commented the **2025-06-09** at **05:52:47**:<br>

OK, I can do that. I had used this command earlier:

`cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF`

and you want me to add this additional parameter here?

`cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1`

let me do that, rebuild and come back here.

---

👤 **ikawrakow** commented the **2025-06-09** at **05:53:12**:<br>

Yes.

---

👤 **ikawrakow** commented the **2025-06-09** at **06:05:31**:<br>

So, I normally don't watch YT. Had to go to another computer as I don't have sound on my development machine. I tried watching the video, but it is constantly being interrupted by advertisements. I think it is better to keep the conversation here. From the log you posted we see that the KV cache is just 600 MB, and the model is taking just ~14 GB. The only thing that we still need to see is the compute buffer size after you rebuild and ran with `-DGGML_SCHED_MAX_COPIES=1`

---

👤 **mtcl** commented the **2025-06-09** at **06:09:53**:<br>

I understand, here is the full log after your suggestion. 

```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1
-- OpenMP found
-- Using optimized iqk matrix multiplications
-- Enabling IQK Flash Attention kernels
-- Using llamafile
-- CUDA found
-- Using CUDA architectures: native
-- CUDA host compiler is GNU 13.3.0

-- Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF
-- CMAKE_SYSTEM_PROCESSOR: x86_64
-- x86 detected
-- ARCH_FLAGS = -march=native
-- Configuring done (0.4s)
-- Generating done (0.1s)
-- Build files have been written to: /home/mukul/dev-ai/ik_llama.cpp/build
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ cmake --build ./build --config Release -j 100
[  0%] Built target build_info
[  0%] Built target sha256
[  1%] Built target xxhash
[  1%] Built target sha1
[  1%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml.c.o
[  2%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-alloc.c.o
[  2%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-backend.c.o
[  2%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-quants.c.o
[  3%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/acc.cu.o
[  3%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/arange.cu.o
[  4%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/argsort.cu.o
[  4%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/binbcast.cu.o
[  4%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/clamp.cu.o
[  5%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/concat.cu.o
[  5%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/conv-transpose-1d.cu.o
[  5%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/convert.cu.o
[  6%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/cpy.cu.o
[  6%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/diagmask.cu.o
[  6%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/dmmv.cu.o
[  7%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/iqk_mmvq.cu.o
[  8%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/fattn-new-mma.cu.o
[  9%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/fattn-tile-f16.cu.o
[  9%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/mmvq.cu.o
[  9%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/fattn-tile-f32.cu.o
[ 10%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/fattn.cu.o
[ 11%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/pool2d.cu.o
[ 11%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/im2col.cu.o
[ 11%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/getrows.cu.o
[ 11%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/quantize.cu.o
[ 11%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/mmq.cu.o
[ 11%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/norm.cu.o
[ 11%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/pad.cu.o
[ 12%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/scale.cu.o
[ 12%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/rope.cu.o
[ 12%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/softcap.cu.o
[ 13%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/sumrows.cu.o
[ 13%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/tsembd.cu.o
[ 13%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/softmax.cu.o
[ 13%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/unary.cu.o
[ 14%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/upscale.cu.o
[ 14%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda.cu.o
[ 15%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-wmma-f16-instance-kqfloat-cpb16.cu.o
[ 15%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-wmma-f16-instance-kqfloat-cpb32.cu.o
[ 15%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-wmma-f16-instance-kqhalf-cpb16.cu.o
[ 16%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-wmma-f16-instance-kqhalf-cpb32.cu.o
[ 16%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-wmma-f16-instance-kqhalf-cpb8.cu.o
[ 16%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_8.cu.o
[ 17%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_1.cu.o
[ 17%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_2.cu.o
[ 17%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_4.cu.o
[ 18%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_4.cu.o
[ 18%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_8.cu.o
[ 18%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_32-ncols2_1.cu.o
[ 19%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_32-ncols2_2.cu.o
[ 19%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_2.cu.o
[ 20%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_8.cu.o
[ 20%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_64-ncols2_1.cu.o
[ 21%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_1.cu.o
[ 21%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_2.cu.o
[ 21%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_4.cu.o
[ 21%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_4.cu.o
[ 22%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_8.cu.o
[ 22%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq1_s.cu.o
[ 23%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq2_k.cu.o
[ 23%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq2_ks.cu.o
[ 23%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq2_s.cu.o
[ 23%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq2_xxs.cu.o
[ 24%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq2_xs.cu.o
[ 24%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq1_s_r4.cu.o
[ 24%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq3_k.cu.o
[ 25%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq3_s.cu.o
[ 26%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq4_k.cu.o
[ 26%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq4_ks.cu.o
[ 26%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq4_ks_r4.cu.o
[ 27%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq4_nl.cu.o
[ 27%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq4_xs.cu.o
[ 27%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq5_k.cu.o
[ 28%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq5_ks.cu.o
[ 28%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq3_xxs.cu.o
[ 28%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq5_ks_r4.cu.o
[ 29%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q2_k.cu.o
[ 29%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q3_k.cu.o
[ 29%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq6_k.cu.o
[ 29%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q4_0.cu.o
[ 30%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q4_1.cu.o
[ 30%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q4_k.cu.o
[ 30%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q5_0.cu.o
[ 31%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q5_1.cu.o
[ 31%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q5_k.cu.o
[ 32%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q6_0.cu.o
[ 32%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q6_k.cu.o
[ 32%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q8_0.cu.o
[ 33%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-q4_0-q4_0.cu.o
[ 33%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-q4_0-q4_0.cu.o
[ 33%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-q8_0-q8_0.cu.o
[ 34%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs192-q8_0-q8_0.cu.o
[ 34%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs256-q8_0-q8_0.cu.o
[ 34%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-q8_0-q8_0.cu.o
[ 35%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs192-q8_0-q8_0.cu.o
[ 35%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs256-q8_0-q8_0.cu.o
[ 35%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-f16-f16.cu.o
[ 36%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs192-f16-f16.cu.o
[ 36%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs256-f16-f16.cu.o
[ 37%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs64-f16-f16.cu.o
[ 37%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-f16-f16.cu.o
[ 37%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs192-f16-f16.cu.o
[ 38%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs256-f16-f16.cu.o
[ 38%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs64-f16-f16.cu.o
[ 38%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-q8_0-iq4_nl.cu.o
[ 39%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-q8_0-iq4_nl.cu.o
[ 39%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-iq4_nl-iq4_nl.cu.o
[ 39%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-iq4_nl-iq4_nl.cu.o
[ 40%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-q6_0-q5_0.cu.o
[ 40%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-q6_0-q5_0.cu.o
[ 40%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-q8_0-q6_0.cu.o
[ 41%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-q8_0-q6_0.cu.o
[ 41%] Building CXX object ggml/src/CMakeFiles/ggml.dir/llamafile/sgemm.cpp.o
[ 41%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_mul_mat.cpp.o
[ 42%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_flash_attn.cpp.o
[ 42%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_576_512.cpp.o
[ 43%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_192_128.cpp.o
[ 43%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_256_256.cpp.o
[ 43%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_128_128.cpp.o
[ 44%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_96_96.cpp.o
[ 44%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_64_64.cpp.o
[ 44%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_floats.cpp.o
[ 45%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_kquants.cpp.o
[ 45%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_ktquants.cpp.o
[ 45%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_iquants.cpp.o
[ 46%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_iqk_quants.cpp.o
[ 46%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_1bit.cpp.o
[ 46%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_legacy_quants.cpp.o
[ 47%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_quantize.cpp.o
[ 47%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-aarch64.c.o
[ 48%] Linking CXX shared library libggml.so
[ 48%] Built target ggml
[ 48%] Linking CXX executable ../../bin/llama-gguf-hash
[ 48%] Linking CXX shared library libllama.so
[ 48%] Linking CXX executable ../../bin/llama-gguf
[ 49%] Built target llama-gguf-hash
[ 50%] Built target llama-gguf
[ 52%] Built target llama
[ 52%] Linking C executable ../bin/test-c
[ 52%] Linking CXX executable ../../bin/llama-quantize-stats
[ 53%] Linking CXX executable ../../bin/llama-bench-matmult
[ 54%] Built target llava
[ 57%] Built target common
[ 58%] Built target llava_static
[ 59%] Linking CXX executable ../bin/test-tokenizer-0
[ 59%] Linking CXX executable ../bin/test-tokenizer-1-bpe
[ 59%] Linking CXX shared library libllava_shared.so
[ 59%] Linking CXX executable ../bin/test-tokenizer-1-spm
[ 59%] Linking CXX executable ../bin/test-chat-template
[ 59%] Linking CXX executable ../bin/test-quantize-perf
[ 60%] Linking CXX executable ../bin/test-model-load-cancel
[ 62%] Linking CXX executable ../bin/test-quantize-fns
[ 62%] Linking CXX executable ../bin/test-grammar-parser
[ 63%] Linking CXX executable ../bin/test-backend-ops
[ 63%] Linking CXX executable ../bin/test-sampling
[ 64%] Linking CXX executable ../../bin/llama-baby-llama
[ 64%] Linking CXX executable ../../bin/llama-convert-llama2c-to-ggml
[ 65%] Linking CXX executable ../bin/test-grammar-integration
[ 66%] Linking CXX executable ../../bin/llama-cvector-generator
[ 67%] Linking CXX executable ../bin/test-autorelease
[ 67%] Built target test-c
[ 67%] Linking CXX executable ../../bin/llama-batched
[ 67%] Linking CXX executable ../../bin/llama-embedding
[ 67%] Linking CXX executable ../../bin/llama-imatrix
[ 68%] Linking CXX executable ../../bin/llama-infill
[ 69%] Linking CXX executable ../../bin/llama-gguf-split
[ 70%] Linking CXX executable ../bin/test-llama-grammar
[ 70%] Linking CXX executable ../bin/test-rope
[ 71%] Linking CXX executable ../../bin/llama-bench
[ 71%] Linking CXX executable ../../bin/llama-batched-bench
[ 72%] Linking CXX executable ../../bin/llama-lookahead
[ 72%] Linking CXX executable ../bin/test-grad0
[ 72%] Linking CXX executable ../../bin/llama-minicpmv-cli
[ 73%] Linking CXX executable ../../bin/llama-export-lora
[ 73%] Linking CXX executable ../../bin/llama-eval-callback
[ 73%] Linking CXX executable ../../bin/llama-gritlm
[ 74%] Linking CXX executable ../bin/test-json-schema-to-grammar
[ 75%] Linking CXX executable ../../bin/llama-lookup-create
[ 75%] Linking CXX executable ../../bin/llama-gbnf-validator
[ 75%] Linking CXX executable ../../bin/llama-lookup-merge
[ 75%] Linking CXX executable ../../bin/llama-parallel
[ 75%] Linking CXX executable ../../bin/llama-lookup
[ 75%] Linking CXX executable ../../bin/llama-llava-cli
[ 75%] Linking CXX executable ../../bin/llama-lookup-stats
[ 75%] Linking CXX executable ../../bin/llama-cli
[ 75%] Linking CXX executable ../../bin/llama-passkey
[ 76%] Linking CXX executable ../../bin/llama-quantize
[ 76%] Linking CXX executable ../../bin/llama-perplexity
[ 77%] Linking CXX executable ../../bin/llama-retrieval
[ 77%] Linking CXX executable ../../bin/llama-speculative
[ 77%] Linking CXX executable ../../bin/llama-sweep-bench
[ 78%] Linking CXX executable ../../bin/llama-simple
[ 78%] Linking CXX executable ../../bin/llama-vdot
[ 79%] Linking CXX executable ../../bin/llama-server
[ 80%] Linking CXX executable ../../bin/llama-q8dot
[ 80%] Linking CXX executable ../../bin/llama-save-load-state
[ 81%] Linking CXX executable ../../bin/llama-tokenize
[ 81%] Built target llama-bench-matmult
[ 82%] Built target llama-quantize-stats
[ 82%] Built target llava_shared
[ 83%] Built target test-grad0
[ 83%] Built target test-quantize-fns
[ 84%] Built target test-autorelease
[ 84%] Built target test-llama-grammar
[ 84%] Built target llama-lookup-merge
[ 85%] Built target llama-gbnf-validator
[ 85%] Built target test-sampling
[ 86%] Built target test-grammar-integration
[ 86%] Built target llama-q8dot
[ 86%] Built target test-grammar-parser
[ 86%] Built target llama-vdot
[ 87%] Built target test-tokenizer-1-spm
[ 87%] Built target test-tokenizer-1-bpe
[ 87%] Built target test-tokenizer-0
[ 88%] Built target test-chat-template
[ 88%] Built target test-json-schema-to-grammar
[ 88%] Built target test-model-load-cancel
[ 88%] Built target llama-cvector-generator
[ 88%] Built target llama-batched
[ 89%] Built target llama-imatrix
[ 89%] Built target llama-minicpmv-cli
[ 90%] Built target llama-batched-bench
[ 90%] Built target llama-infill
[ 90%] Built target llama-gritlm
[ 92%] Built target llama-eval-callback
[ 92%] Built target llama-lookahead
[ 93%] Built target llama-lookup-stats
[ 94%] Built target llama-convert-llama2c-to-ggml
[ 94%] Built target llama-retrieval
[ 94%] Built target llama-bench
[ 94%] Built target llama-llava-cli
[ 94%] Built target llama-parallel
[ 94%] Built target llama-export-lora
[ 95%] Built target llama-passkey
[ 95%] Built target llama-cli
[ 95%] Built target llama-speculative
[ 95%] Built target llama-save-load-state
[ 95%] Built target llama-gguf-split
[ 95%] Built target test-backend-ops
[ 95%] Built target llama-tokenize
[ 95%] Built target llama-simple
[ 95%] Built target llama-embedding
[ 96%] Built target llama-server
[ 96%] Built target llama-lookup-create
[ 96%] Built target llama-perplexity
[ 97%] Built target llama-lookup
[ 98%] Built target test-rope
[ 99%] Built target test-quantize-perf
[100%] Built target llama-sweep-bench
[100%] Built target llama-quantize
[100%] Built target llama-baby-llama
```

```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-server \
    --model /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf \
    --alias ubergarm/DeepSeek-R1-0528-GGUF \
    --ctx-size 32768 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 2048 -ub 2048 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    -ot "blk\.(3)\.ffn_.*=CUDA0" \
    -ot "blk\.(5)\.ffn_.*=CUDA1" \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```

```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
INFO [                    main] build info | tid="124625576644608" timestamp=1749449241 build=3737 commit="58f08e43"
INFO [                    main] system info | tid="124625576644608" timestamp=1749449241 n_threads=57 n_threads_batch=-1 total_threads=112 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 6 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 52 key-value pairs and 1147 tensors from /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 0528
llama_model_loader: - kv   3:                            general.version str              = 0528
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek-R1
llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
llama_model_loader: - kv   6:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   7:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   8:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   9:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  10:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  11:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  12:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  13: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  15:                          general.file_type u32              = 339
llama_model_loader: - kv  16:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  17:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  18:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  19:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  20:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  21:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  22:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  23:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  24:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  25:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  26:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  27:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  28:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  29:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  30:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  31: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  32: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  33:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  34:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  36:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  37:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  38:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  39:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  40:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  41:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  42:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  43:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  44:               general.quantization_version u32              = 2
llama_model_loader: - kv  45:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/DeepSeek-R1...
llama_model_loader: - kv  46:                   quantize.imatrix.dataset str              = ubergarm-imatrix-calibration-corpus-v...
llama_model_loader: - kv  47:             quantize.imatrix.entries_count i32              = 721
llama_model_loader: - kv  48:              quantize.imatrix.chunks_count i32              = 812
llama_model_loader: - kv  49:                                   split.no u16              = 0
llama_model_loader: - kv  50:                                split.count u16              = 7
llama_model_loader: - kv  51:                        split.tensors.count i32              = 1147
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq3_k_r4:  116 tensors
llama_model_loader: - type iq4_ks_r4:   58 tensors
llm_load_vocab: special tokens cache size = 818
llm_load_vocab: token to piece cache size = 0.8223 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 129280
llm_load_print_meta: n_merges         = 127741
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 163840
llm_load_print_meta: n_embd           = 7168
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 128
llm_load_print_meta: n_head_kv        = 128
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 24576
llm_load_print_meta: n_embd_v_gqa     = 16384
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 18432
llm_load_print_meta: n_expert         = 256
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 0.025
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = IQ3_K_R4 - 3.4325 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 300.938 GiB (3.847 BPW) 
llm_load_print_meta: repeating layers = 299.104 GiB (3.834 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1 0528
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: LF token         = 131 'Ä'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_layer_dense_lead   = 3
llm_load_print_meta: n_lora_q             = 1536
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 2048
llm_load_print_meta: n_expert_shared      = 1
llm_load_print_meta: expert_weights_scale = 2.5
llm_load_print_meta: expert_weights_norm  = 1
llm_load_print_meta: expert_gating_func   = sigmoid
llm_load_print_meta: rope_yarn_log_mul    = 0.1000
llm_load_tensors: ggml ctx size =    1.40 MiB
Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 36486.67 MiB
llm_load_tensors:        CPU buffer size = 43905.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 43905.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 44473.21 MiB
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 13995.99 MiB
llm_load_tensors:      CUDA1 buffer size = 13730.03 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 2048
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =   592.89 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   573.76 MiB
llama_new_context_with_model: KV self size  = 1166.62 MiB, c^KV (q8_0): 1166.62 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  3588.01 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  3560.02 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   312.02 MiB
llama_new_context_with_model: graph nodes  = 8245
llama_new_context_with_model: graph splits = 149
INFO [                    init] initializing slots | tid="124625576644608" timestamp=1749449287 n_slots=1
INFO [                    init] new slot | tid="124625576644608" timestamp=1749449287 id_slot=0 n_ctx_slot=32768
INFO [                    main] model loaded | tid="124625576644608" timestamp=1749449287
INFO [                    main] chat template | tid="124625576644608" timestamp=1749449287 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
INFO [                    main] HTTP server listening | tid="124625576644608" timestamp=1749449287 n_threads_http="111" port="10002" hostname="0.0.0.0"
```
```
INFO [            update_slots] all slots are idle | tid="124625576644608" timestamp=1749449287
INFO [      log_server_request] request | tid="124623467307008" timestamp=1749449303 remote_addr="172.17.0.3" remote_port=41390 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="124623458914304" timestamp=1749449310 remote_addr="172.17.0.3" remote_port=41406 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="124623375036416" timestamp=1749449312 remote_addr="172.17.0.3" remote_port=50732 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="124623366643712" timestamp=1749449314 remote_addr="172.17.0.3" remote_port=50738 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="124625576644608" timestamp=1749449314 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="124625576644608" timestamp=1749449314 id_slot=0 id_task=0 p0=0
INFO [           print_timings] prompt eval time     =     373.61 ms /     4 tokens (   93.40 ms per token,    10.71 tokens per second) | tid="124625576644608" timestamp=1749449333 id_slot=0 id_task=0 t_prompt_processing=373.606 n_prompt_tokens_processed=4 t_token=93.4015 n_tokens_second=10.70646617024352
INFO [           print_timings] generation eval time =   17732.40 ms /   181 runs   (   97.97 ms per token,    10.21 tokens per second) | tid="124625576644608" timestamp=1749449333 id_slot=0 id_task=0 t_token_generation=17732.405 n_decoded=181 t_token=97.96908839779005 n_tokens_second=10.20730126567716
INFO [           print_timings]           total time =   18106.01 ms | tid="124625576644608" timestamp=1749449333 id_slot=0 id_task=0 t_prompt_processing=373.606 t_token_generation=17732.405 t_total=18106.011
INFO [            update_slots] slot released | tid="124625576644608" timestamp=1749449333 id_slot=0 id_task=0 n_ctx=32768 n_past=184 n_system_tokens=0 n_cache_tokens=184 truncated=false
INFO [            update_slots] all slots are idle | tid="124625576644608" timestamp=1749449333
INFO [      log_server_request] request | tid="124623358251008" timestamp=1749449333 remote_addr="172.17.0.3" remote_port=50750 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="124625576644608" timestamp=1749449333
```

---

👤 **mtcl** commented the **2025-06-09** at **06:09:53**:<br>

I understand, here is the full log after your suggestion. 

(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1
-- OpenMP found
-- Using optimized iqk matrix multiplications
-- Enabling IQK Flash Attention kernels
-- Using llamafile
-- CUDA found
-- Using CUDA architectures: native
-- CUDA host compiler is GNU 13.3.0

-- Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF
-- CMAKE_SYSTEM_PROCESSOR: x86_64
-- x86 detected
-- ARCH_FLAGS = -march=native
-- Configuring done (0.4s)
-- Generating done (0.1s)
-- Build files have been written to: /home/mukul/dev-ai/ik_llama.cpp/build
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ cmake --build ./build --config Release -j 100
[  0%] Built target build_info
[  0%] Built target sha256
[  1%] Built target xxhash
[  1%] Built target sha1
[  1%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml.c.o
[  2%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-alloc.c.o
[  2%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-backend.c.o
[  2%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-quants.c.o
[  3%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/acc.cu.o
[  3%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/arange.cu.o
[  4%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/argsort.cu.o
[  4%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/binbcast.cu.o
[  4%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/clamp.cu.o
[  5%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/concat.cu.o
[  5%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/conv-transpose-1d.cu.o
[  5%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/convert.cu.o
[  6%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/cpy.cu.o
[  6%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/diagmask.cu.o
[  6%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/dmmv.cu.o
[  7%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/iqk_mmvq.cu.o
[  8%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/fattn-new-mma.cu.o
[  9%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/fattn-tile-f16.cu.o
[  9%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/mmvq.cu.o
[  9%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/fattn-tile-f32.cu.o
[ 10%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/fattn.cu.o
[ 11%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/pool2d.cu.o
[ 11%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/im2col.cu.o
[ 11%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/getrows.cu.o
[ 11%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/quantize.cu.o
[ 11%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/mmq.cu.o
[ 11%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/norm.cu.o
[ 11%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/pad.cu.o
[ 12%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/scale.cu.o
[ 12%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/rope.cu.o
[ 12%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/softcap.cu.o
[ 13%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/sumrows.cu.o
[ 13%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/tsembd.cu.o
[ 13%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/softmax.cu.o
[ 13%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/unary.cu.o
[ 14%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/upscale.cu.o
[ 14%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda.cu.o
[ 15%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-wmma-f16-instance-kqfloat-cpb16.cu.o
[ 15%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-wmma-f16-instance-kqfloat-cpb32.cu.o
[ 15%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-wmma-f16-instance-kqhalf-cpb16.cu.o
[ 16%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-wmma-f16-instance-kqhalf-cpb32.cu.o
[ 16%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-wmma-f16-instance-kqhalf-cpb8.cu.o
[ 16%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_8.cu.o
[ 17%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_1.cu.o
[ 17%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_2.cu.o
[ 17%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_4.cu.o
[ 18%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_4.cu.o
[ 18%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_8.cu.o
[ 18%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_32-ncols2_1.cu.o
[ 19%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_32-ncols2_2.cu.o
[ 19%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_2.cu.o
[ 20%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_8.cu.o
[ 20%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_64-ncols2_1.cu.o
[ 21%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_1.cu.o
[ 21%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_2.cu.o
[ 21%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_4.cu.o
[ 21%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_4.cu.o
[ 22%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_8.cu.o
[ 22%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq1_s.cu.o
[ 23%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq2_k.cu.o
[ 23%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq2_ks.cu.o
[ 23%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq2_s.cu.o
[ 23%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq2_xxs.cu.o
[ 24%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq2_xs.cu.o
[ 24%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq1_s_r4.cu.o
[ 24%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq3_k.cu.o
[ 25%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq3_s.cu.o
[ 26%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq4_k.cu.o
[ 26%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq4_ks.cu.o
[ 26%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq4_ks_r4.cu.o
[ 27%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq4_nl.cu.o
[ 27%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq4_xs.cu.o
[ 27%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq5_k.cu.o
[ 28%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq5_ks.cu.o
[ 28%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq3_xxs.cu.o
[ 28%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq5_ks_r4.cu.o
[ 29%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q2_k.cu.o
[ 29%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q3_k.cu.o
[ 29%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq6_k.cu.o
[ 29%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q4_0.cu.o
[ 30%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q4_1.cu.o
[ 30%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q4_k.cu.o
[ 30%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q5_0.cu.o
[ 31%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q5_1.cu.o
[ 31%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q5_k.cu.o
[ 32%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q6_0.cu.o
[ 32%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q6_k.cu.o
[ 32%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q8_0.cu.o
[ 33%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-q4_0-q4_0.cu.o
[ 33%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-q4_0-q4_0.cu.o
[ 33%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-q8_0-q8_0.cu.o
[ 34%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs192-q8_0-q8_0.cu.o
[ 34%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs256-q8_0-q8_0.cu.o
[ 34%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-q8_0-q8_0.cu.o
[ 35%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs192-q8_0-q8_0.cu.o
[ 35%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs256-q8_0-q8_0.cu.o
[ 35%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-f16-f16.cu.o
[ 36%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs192-f16-f16.cu.o
[ 36%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs256-f16-f16.cu.o
[ 37%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs64-f16-f16.cu.o
[ 37%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-f16-f16.cu.o
[ 37%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs192-f16-f16.cu.o
[ 38%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs256-f16-f16.cu.o
[ 38%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs64-f16-f16.cu.o
[ 38%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-q8_0-iq4_nl.cu.o
[ 39%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-q8_0-iq4_nl.cu.o
[ 39%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-iq4_nl-iq4_nl.cu.o
[ 39%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-iq4_nl-iq4_nl.cu.o
[ 40%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-q6_0-q5_0.cu.o
[ 40%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-q6_0-q5_0.cu.o
[ 40%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-q8_0-q6_0.cu.o
[ 41%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-q8_0-q6_0.cu.o
[ 41%] Building CXX object ggml/src/CMakeFiles/ggml.dir/llamafile/sgemm.cpp.o
[ 41%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_mul_mat.cpp.o
[ 42%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_flash_attn.cpp.o
[ 42%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_576_512.cpp.o
[ 43%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_192_128.cpp.o
[ 43%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_256_256.cpp.o
[ 43%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_128_128.cpp.o
[ 44%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_96_96.cpp.o
[ 44%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_64_64.cpp.o
[ 44%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_floats.cpp.o
[ 45%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_kquants.cpp.o
[ 45%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_ktquants.cpp.o
[ 45%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_iquants.cpp.o
[ 46%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_iqk_quants.cpp.o
[ 46%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_1bit.cpp.o
[ 46%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_legacy_quants.cpp.o
[ 47%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_quantize.cpp.o
[ 47%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-aarch64.c.o
[ 48%] Linking CXX shared library libggml.so
[ 48%] Built target ggml
[ 48%] Linking CXX executable ../../bin/llama-gguf-hash
[ 48%] Linking CXX shared library libllama.so
[ 48%] Linking CXX executable ../../bin/llama-gguf
[ 49%] Built target llama-gguf-hash
[ 50%] Built target llama-gguf
[ 52%] Built target llama
[ 52%] Linking C executable ../bin/test-c
[ 52%] Linking CXX executable ../../bin/llama-quantize-stats
[ 53%] Linking CXX executable ../../bin/llama-bench-matmult
[ 54%] Built target llava
[ 57%] Built target common
[ 58%] Built target llava_static
[ 59%] Linking CXX executable ../bin/test-tokenizer-0
[ 59%] Linking CXX executable ../bin/test-tokenizer-1-bpe
[ 59%] Linking CXX shared library libllava_shared.so
[ 59%] Linking CXX executable ../bin/test-tokenizer-1-spm
[ 59%] Linking CXX executable ../bin/test-chat-template
[ 59%] Linking CXX executable ../bin/test-quantize-perf
[ 60%] Linking CXX executable ../bin/test-model-load-cancel
[ 62%] Linking CXX executable ../bin/test-quantize-fns
[ 62%] Linking CXX executable ../bin/test-grammar-parser
[ 63%] Linking CXX executable ../bin/test-backend-ops
[ 63%] Linking CXX executable ../bin/test-sampling
[ 64%] Linking CXX executable ../../bin/llama-baby-llama
[ 64%] Linking CXX executable ../../bin/llama-convert-llama2c-to-ggml
[ 65%] Linking CXX executable ../bin/test-grammar-integration
[ 66%] Linking CXX executable ../../bin/llama-cvector-generator
[ 67%] Linking CXX executable ../bin/test-autorelease
[ 67%] Built target test-c
[ 67%] Linking CXX executable ../../bin/llama-batched
[ 67%] Linking CXX executable ../../bin/llama-embedding
[ 67%] Linking CXX executable ../../bin/llama-imatrix
[ 68%] Linking CXX executable ../../bin/llama-infill
[ 69%] Linking CXX executable ../../bin/llama-gguf-split
[ 70%] Linking CXX executable ../bin/test-llama-grammar
[ 70%] Linking CXX executable ../bin/test-rope
[ 71%] Linking CXX executable ../../bin/llama-bench
[ 71%] Linking CXX executable ../../bin/llama-batched-bench
[ 72%] Linking CXX executable ../../bin/llama-lookahead
[ 72%] Linking CXX executable ../bin/test-grad0
[ 72%] Linking CXX executable ../../bin/llama-minicpmv-cli
[ 73%] Linking CXX executable ../../bin/llama-export-lora
[ 73%] Linking CXX executable ../../bin/llama-eval-callback
[ 73%] Linking CXX executable ../../bin/llama-gritlm
[ 74%] Linking CXX executable ../bin/test-json-schema-to-grammar
[ 75%] Linking CXX executable ../../bin/llama-lookup-create
[ 75%] Linking CXX executable ../../bin/llama-gbnf-validator
[ 75%] Linking CXX executable ../../bin/llama-lookup-merge
[ 75%] Linking CXX executable ../../bin/llama-parallel
[ 75%] Linking CXX executable ../../bin/llama-lookup
[ 75%] Linking CXX executable ../../bin/llama-llava-cli
[ 75%] Linking CXX executable ../../bin/llama-lookup-stats
[ 75%] Linking CXX executable ../../bin/llama-cli
[ 75%] Linking CXX executable ../../bin/llama-passkey
[ 76%] Linking CXX executable ../../bin/llama-quantize
[ 76%] Linking CXX executable ../../bin/llama-perplexity
[ 77%] Linking CXX executable ../../bin/llama-retrieval
[ 77%] Linking CXX executable ../../bin/llama-speculative
[ 77%] Linking CXX executable ../../bin/llama-sweep-bench
[ 78%] Linking CXX executable ../../bin/llama-simple
[ 78%] Linking CXX executable ../../bin/llama-vdot
[ 79%] Linking CXX executable ../../bin/llama-server
[ 80%] Linking CXX executable ../../bin/llama-q8dot
[ 80%] Linking CXX executable ../../bin/llama-save-load-state
[ 81%] Linking CXX executable ../../bin/llama-tokenize
[ 81%] Built target llama-bench-matmult
[ 82%] Built target llama-quantize-stats
[ 82%] Built target llava_shared
[ 83%] Built target test-grad0
[ 83%] Built target test-quantize-fns
[ 84%] Built target test-autorelease
[ 84%] Built target test-llama-grammar
[ 84%] Built target llama-lookup-merge
[ 85%] Built target llama-gbnf-validator
[ 85%] Built target test-sampling
[ 86%] Built target test-grammar-integration
[ 86%] Built target llama-q8dot
[ 86%] Built target test-grammar-parser
[ 86%] Built target llama-vdot
[ 87%] Built target test-tokenizer-1-spm
[ 87%] Built target test-tokenizer-1-bpe
[ 87%] Built target test-tokenizer-0
[ 88%] Built target test-chat-template
[ 88%] Built target test-json-schema-to-grammar
[ 88%] Built target test-model-load-cancel
[ 88%] Built target llama-cvector-generator
[ 88%] Built target llama-batched
[ 89%] Built target llama-imatrix
[ 89%] Built target llama-minicpmv-cli
[ 90%] Built target llama-batched-bench
[ 90%] Built target llama-infill
[ 90%] Built target llama-gritlm
[ 92%] Built target llama-eval-callback
[ 92%] Built target llama-lookahead
[ 93%] Built target llama-lookup-stats
[ 94%] Built target llama-convert-llama2c-to-ggml
[ 94%] Built target llama-retrieval
[ 94%] Built target llama-bench
[ 94%] Built target llama-llava-cli
[ 94%] Built target llama-parallel
[ 94%] Built target llama-export-lora
[ 95%] Built target llama-passkey
[ 95%] Built target llama-cli
[ 95%] Built target llama-speculative
[ 95%] Built target llama-save-load-state
[ 95%] Built target llama-gguf-split
[ 95%] Built target test-backend-ops
[ 95%] Built target llama-tokenize
[ 95%] Built target llama-simple
[ 95%] Built target llama-embedding
[ 96%] Built target llama-server
[ 96%] Built target llama-lookup-create
[ 96%] Built target llama-perplexity
[ 97%] Built target llama-lookup
[ 98%] Built target test-rope
[ 99%] Built target test-quantize-perf
[100%] Built target llama-sweep-bench
[100%] Built target llama-quantize
[100%] Built target llama-baby-llama
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-server \
    --model /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf \
    --alias ubergarm/DeepSeek-R1-0528-GGUF \
    --ctx-size 32768 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 2048 -ub 2048 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    -ot "blk\.(3)\.ffn_.*=CUDA0" \
    -ot "blk\.(5)\.ffn_.*=CUDA1" \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
INFO [                    main] build info | tid="124625576644608" timestamp=1749449241 build=3737 commit="58f08e43"
INFO [                    main] system info | tid="124625576644608" timestamp=1749449241 n_threads=57 n_threads_batch=-1 total_threads=112 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 6 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 52 key-value pairs and 1147 tensors from /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 0528
llama_model_loader: - kv   3:                            general.version str              = 0528
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek-R1
llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
llama_model_loader: - kv   6:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   7:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   8:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   9:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  10:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  11:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  12:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  13: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  15:                          general.file_type u32              = 339
llama_model_loader: - kv  16:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  17:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  18:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  19:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  20:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  21:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  22:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  23:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  24:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  25:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  26:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  27:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  28:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  29:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  30:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  31: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  32: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  33:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  34:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  36:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  37:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  38:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  39:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  40:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  41:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  42:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  43:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  44:               general.quantization_version u32              = 2
llama_model_loader: - kv  45:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/DeepSeek-R1...
llama_model_loader: - kv  46:                   quantize.imatrix.dataset str              = ubergarm-imatrix-calibration-corpus-v...
llama_model_loader: - kv  47:             quantize.imatrix.entries_count i32              = 721
llama_model_loader: - kv  48:              quantize.imatrix.chunks_count i32              = 812
llama_model_loader: - kv  49:                                   split.no u16              = 0
llama_model_loader: - kv  50:                                split.count u16              = 7
llama_model_loader: - kv  51:                        split.tensors.count i32              = 1147
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq3_k_r4:  116 tensors
llama_model_loader: - type iq4_ks_r4:   58 tensors
llm_load_vocab: special tokens cache size = 818
llm_load_vocab: token to piece cache size = 0.8223 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 129280
llm_load_print_meta: n_merges         = 127741
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 163840
llm_load_print_meta: n_embd           = 7168
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 128
llm_load_print_meta: n_head_kv        = 128
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 24576
llm_load_print_meta: n_embd_v_gqa     = 16384
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 18432
llm_load_print_meta: n_expert         = 256
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 0.025
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = IQ3_K_R4 - 3.4325 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 300.938 GiB (3.847 BPW) 
llm_load_print_meta: repeating layers = 299.104 GiB (3.834 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1 0528
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: LF token         = 131 'Ä'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_layer_dense_lead   = 3
llm_load_print_meta: n_lora_q             = 1536
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 2048
llm_load_print_meta: n_expert_shared      = 1
llm_load_print_meta: expert_weights_scale = 2.5
llm_load_print_meta: expert_weights_norm  = 1
llm_load_print_meta: expert_gating_func   = sigmoid
llm_load_print_meta: rope_yarn_log_mul    = 0.1000
llm_load_tensors: ggml ctx size =    1.40 MiB
Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 36486.67 MiB
llm_load_tensors:        CPU buffer size = 43905.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 43905.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 44473.21 MiB
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 13995.99 MiB
llm_load_tensors:      CUDA1 buffer size = 13730.03 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 2048
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =   592.89 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   573.76 MiB
llama_new_context_with_model: KV self size  = 1166.62 MiB, c^KV (q8_0): 1166.62 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  3588.01 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  3560.02 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   312.02 MiB
llama_new_context_with_model: graph nodes  = 8245
llama_new_context_with_model: graph splits = 149
INFO [                    init] initializing slots | tid="124625576644608" timestamp=1749449287 n_slots=1
INFO [                    init] new slot | tid="124625576644608" timestamp=1749449287 id_slot=0 n_ctx_slot=32768
INFO [                    main] model loaded | tid="124625576644608" timestamp=1749449287
INFO [                    main] chat template | tid="124625576644608" timestamp=1749449287 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
INFO [                    main] HTTP server listening | tid="124625576644608" timestamp=1749449287 n_threads_http="111" port="10002" hostname="0.0.0.0"
INFO [            update_slots] all slots are idle | tid="124625576644608" timestamp=1749449287
INFO [      log_server_request] request | tid="124623467307008" timestamp=1749449303 remote_addr="172.17.0.3" remote_port=41390 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="124623458914304" timestamp=1749449310 remote_addr="172.17.0.3" remote_port=41406 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="124623375036416" timestamp=1749449312 remote_addr="172.17.0.3" remote_port=50732 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="124623366643712" timestamp=1749449314 remote_addr="172.17.0.3" remote_port=50738 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="124625576644608" timestamp=1749449314 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="124625576644608" timestamp=1749449314 id_slot=0 id_task=0 p0=0
INFO [           print_timings] prompt eval time     =     373.61 ms /     4 tokens (   93.40 ms per token,    10.71 tokens per second) | tid="124625576644608" timestamp=1749449333 id_slot=0 id_task=0 t_prompt_processing=373.606 n_prompt_tokens_processed=4 t_token=93.4015 n_tokens_second=10.70646617024352
INFO [           print_timings] generation eval time =   17732.40 ms /   181 runs   (   97.97 ms per token,    10.21 tokens per second) | tid="124625576644608" timestamp=1749449333 id_slot=0 id_task=0 t_token_generation=17732.405 n_decoded=181 t_token=97.96908839779005 n_tokens_second=10.20730126567716
INFO [           print_timings]           total time =   18106.01 ms | tid="124625576644608" timestamp=1749449333 id_slot=0 id_task=0 t_prompt_processing=373.606 t_token_generation=17732.405 t_total=18106.011
INFO [            update_slots] slot released | tid="124625576644608" timestamp=1749449333 id_slot=0 id_task=0 n_ctx=32768 n_past=184 n_system_tokens=0 n_cache_tokens=184 truncated=false
INFO [            update_slots] all slots are idle | tid="124625576644608" timestamp=1749449333
INFO [      log_server_request] request | tid="124623358251008" timestamp=1749449333 remote_addr="172.17.0.3" remote_port=50750 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="124625576644608" timestamp=1749449333

---

👤 **mtcl** commented the **2025-06-09** at **06:10:23**:<br>

I have about 18GB on both GPUs now.

---

👤 **ikawrakow** commented the **2025-06-09** at **06:14:03**:<br>

OK, first try `-b 4096 -ub 4096` to see if this will fit. If it fits, you will give you a much better prefill if you are processing long contexts.

---

👤 **ikawrakow** commented the **2025-06-09** at **06:15:16**:<br>

OK, with 4 tokens of prompt nobody can get more than 10 t/s prefill. You need to try a few thousand tokens prompt (that's when the prefill speed starts to matter).

---

👤 **mtcl** commented the **2025-06-09** at **06:16:51**:<br>

Oh yes, i just tried a hi, i understand that 10tk/sec is not accurate for such a short prompt. I am running it on about 16k tokens now and i will report back. after that i will modify this and send same back to the model.

```bash
CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-server \
    --model /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf \
    --alias ubergarm/DeepSeek-R1-0528-GGUF \
    --ctx-size 32768 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 4096 -ub 4096 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    -ot "blk\.(3)\.ffn_.*=CUDA0" \
    -ot "blk\.(5)\.ffn_.*=CUDA1" \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```

---

👤 **saood06** commented the **2025-06-09** at **06:19:38**:<br>

> Try `cmake -DGGML_SCHED_MAX_COPIES=1 ...`

I keep forgetting to mention this to you, but I think the reason people keep needing to set this is pipeline parallelism checks for whether the model is fully offloaded by `model->n_gpu_layers > (int)model->hparams.n_layer` and with tensor offload that assumption is no longer true. So just adding a check for if `override-tensor` is used and not enabling it would solve the issue (and I think that is what mainline did from my memory)

---

👤 **mtcl** commented the **2025-06-09** at **06:26:03**:<br>

> OK, with 4 tokens of prompt nobody can get more than 10 t/s prefill. You need to try a few thousand tokens prompt (that's when the prefill speed starts to matter).

OK here is the 16K context processing:

```
INFO [   launch_slot_with_task] slot is processing task | tid="124625576644608" timestamp=1749449891 id_slot=0 id_task=1558
INFO [            update_slots] kv cache rm [p0, end) | tid="124625576644608" timestamp=1749449891 id_slot=0 id_task=1558 p0=2





INFO [      log_server_request] request | tid="124623282716672" timestamp=1749449897 remote_addr="172.17.0.3" remote_port=46266 status=200 method="GET" path="/v1/models" params={}
INFO [            update_slots] kv cache rm [p0, end) | tid="124625576644608" timestamp=1749449924 id_slot=0 id_task=1558 p0=2050
INFO [            update_slots] kv cache rm [p0, end) | tid="124625576644608" timestamp=1749449957 id_slot=0 id_task=1558 p0=4098
INFO [            update_slots] kv cache rm [p0, end) | tid="124625576644608" timestamp=1749449990 id_slot=0 id_task=1558 p0=6146
INFO [            update_slots] kv cache rm [p0, end) | tid="124625576644608" timestamp=1749450024 id_slot=0 id_task=1558 p0=8194
INFO [            update_slots] kv cache rm [p0, end) | tid="124625576644608" timestamp=1749450057 id_slot=0 id_task=1558 p0=10242
INFO [            update_slots] kv cache rm [p0, end) | tid="124625576644608" timestamp=1749450091 id_slot=0 id_task=1558 p0=12290
INFO [            update_slots] kv cache rm [p0, end) | tid="124625576644608" timestamp=1749450125 id_slot=0 id_task=1558 p0=14338
INFO [      log_server_request] request | tid="124623274323968" timestamp=1749450145 remote_addr="172.17.0.3" remote_port=33860 status=200 method="GET" path="/v1/models" params={}
INFO [            update_slots] kv cache rm [p0, end) | tid="124625576644608" timestamp=1749450159 id_slot=0 id_task=1558 p0=16386
INFO [            update_slots] kv cache rm [p0, end) | tid="124625576644608" timestamp=1749450194 id_slot=0 id_task=1558 p0=18434
INFO [           print_timings] prompt eval time     =  336582.34 ms / 19383 tokens (   17.36 ms per token,    57.59 tokens per second) | tid="124625576644608" timestamp=1749450270 id_slot=0 id_task=1558 t_prompt_processing=336582.34 n_prompt_tokens_processed=19383 t_token=17.364821751018937 n_tokens_second=57.58769161804508
INFO [           print_timings] generation eval time =   42214.69 ms /   388 runs   (  108.80 ms per token,     9.19 tokens per second) | tid="124625576644608" timestamp=1749450270 id_slot=0 id_task=1558 t_token_generation=42214.691 n_decoded=388 t_token=108.80075 n_tokens_second=9.191113112731301
INFO [           print_timings]           total time =  378797.03 ms | tid="124625576644608" timestamp=1749450270 id_slot=0 id_task=1558 t_prompt_processing=336582.34 t_token_generation=42214.691 t_total=378797.031
INFO [            update_slots] slot released | tid="124625576644608" timestamp=1749450270 id_slot=0 id_task=1558 n_ctx=32768 n_past=19772 n_system_tokens=0 n_cache_tokens=19772 truncated=false
INFO [            update_slots] all slots are idle | tid="124625576644608" timestamp=1749450270
INFO [      log_server_request] request | tid="124623291109376" timestamp=1749450270 remote_addr="172.17.0.3" remote_port=46258 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="124625576644608" timestamp=1749450270


```

i will now use below and try:

```bash
CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-server \
    --model /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf \
    --alias ubergarm/DeepSeek-R1-0528-GGUF \
    --ctx-size 32768 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 4096 -ub 4096 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    -ot "blk\.(3)\.ffn_.*=CUDA0" \
    -ot "blk\.(5)\.ffn_.*=CUDA1" \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```

---

👤 **mtcl** commented the **2025-06-09** at **06:26:03**:<br>

> OK, with 4 tokens of prompt nobody can get more than 10 t/s prefill. You need to try a few thousand tokens prompt (that's when the prefill speed starts to matter).

OK here is the 16K context processing:

```
INFO [   launch_slot_with_task] slot is processing task | tid="124625576644608" timestamp=1749449891 id_slot=0 id_task=1558
INFO [            update_slots] kv cache rm [p0, end) | tid="124625576644608" timestamp=1749449891 id_slot=0 id_task=1558 p0=2





INFO [      log_server_request] request | tid="124623282716672" timestamp=1749449897 remote_addr="172.17.0.3" remote_port=46266 status=200 method="GET" path="/v1/models" params={}
INFO [            update_slots] kv cache rm [p0, end) | tid="124625576644608" timestamp=1749449924 id_slot=0 id_task=1558 p0=2050
INFO [            update_slots] kv cache rm [p0, end) | tid="124625576644608" timestamp=1749449957 id_slot=0 id_task=1558 p0=4098
INFO [            update_slots] kv cache rm [p0, end) | tid="124625576644608" timestamp=1749449990 id_slot=0 id_task=1558 p0=6146
INFO [            update_slots] kv cache rm [p0, end) | tid="124625576644608" timestamp=1749450024 id_slot=0 id_task=1558 p0=8194
INFO [            update_slots] kv cache rm [p0, end) | tid="124625576644608" timestamp=1749450057 id_slot=0 id_task=1558 p0=10242
INFO [            update_slots] kv cache rm [p0, end) | tid="124625576644608" timestamp=1749450091 id_slot=0 id_task=1558 p0=12290
INFO [            update_slots] kv cache rm [p0, end) | tid="124625576644608" timestamp=1749450125 id_slot=0 id_task=1558 p0=14338
INFO [      log_server_request] request | tid="124623274323968" timestamp=1749450145 remote_addr="172.17.0.3" remote_port=33860 status=200 method="GET" path="/v1/models" params={}
INFO [            update_slots] kv cache rm [p0, end) | tid="124625576644608" timestamp=1749450159 id_slot=0 id_task=1558 p0=16386
INFO [            update_slots] kv cache rm [p0, end) | tid="124625576644608" timestamp=1749450194 id_slot=0 id_task=1558 p0=18434
INFO [           print_timings] prompt eval time     =  336582.34 ms / 19383 tokens (   17.36 ms per token,    57.59 tokens per second) | tid="124625576644608" timestamp=1749450270 id_slot=0 id_task=1558 t_prompt_processing=336582.34 n_prompt_tokens_processed=19383 t_token=17.364821751018937 n_tokens_second=57.58769161804508
INFO [           print_timings] generation eval time =   42214.69 ms /   388 runs   (  108.80 ms per token,     9.19 tokens per second) | tid="124625576644608" timestamp=1749450270 id_slot=0 id_task=1558 t_token_generation=42214.691 n_decoded=388 t_token=108.80075 n_tokens_second=9.191113112731301
INFO [           print_timings]           total time =  378797.03 ms | tid="124625576644608" timestamp=1749450270 id_slot=0 id_task=1558 t_prompt_processing=336582.34 t_token_generation=42214.691 t_total=378797.031
INFO [            update_slots] slot released | tid="124625576644608" timestamp=1749450270 id_slot=0 id_task=1558 n_ctx=32768 n_past=19772 n_system_tokens=0 n_cache_tokens=19772 truncated=false
INFO [            update_slots] all slots are idle | tid="124625576644608" timestamp=1749450270
INFO [      log_server_request] request | tid="124623291109376" timestamp=1749450270 remote_addr="172.17.0.3" remote_port=46258 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="124625576644608" timestamp=1749450270


```

i will now use below and try:

```bash
CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-server \
    --model /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf \
    --alias ubergarm/DeepSeek-R1-0528-GGUF \
    --ctx-size 32768 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 4096 -ub 4096 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 20 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```

---

👤 **mtcl** commented the **2025-06-09** at **06:35:12**:<br>

```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-server \
    --model /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf \
    --alias ubergarm/DeepSeek-R1-0528-GGUF \
    --ctx-size 32768 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 4096 -ub 4096 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    -ot "blk\.(3)\.ffn_.*=CUDA0" \
    -ot "blk\.(5)\.ffn_.*=CUDA1" \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```
```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
INFO [                    main] build info | tid="132484236767232" timestamp=1749450548 build=3737 commit="58f08e43"
INFO [                    main] system info | tid="132484236767232" timestamp=1749450548 n_threads=57 n_threads_batch=-1 total_threads=112 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 6 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 52 key-value pairs and 1147 tensors from /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 0528
llama_model_loader: - kv   3:                            general.version str              = 0528
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek-R1
llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
llama_model_loader: - kv   6:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   7:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   8:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   9:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  10:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  11:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  12:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  13: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  15:                          general.file_type u32              = 339
llama_model_loader: - kv  16:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  17:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  18:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  19:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  20:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  21:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  22:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  23:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  24:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  25:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  26:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  27:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  28:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  29:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  30:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  31: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  32: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  33:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  34:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  36:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  37:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  38:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  39:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  40:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  41:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  42:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  43:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  44:               general.quantization_version u32              = 2
llama_model_loader: - kv  45:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/DeepSeek-R1...
llama_model_loader: - kv  46:                   quantize.imatrix.dataset str              = ubergarm-imatrix-calibration-corpus-v...
llama_model_loader: - kv  47:             quantize.imatrix.entries_count i32              = 721
llama_model_loader: - kv  48:              quantize.imatrix.chunks_count i32              = 812
llama_model_loader: - kv  49:                                   split.no u16              = 0
llama_model_loader: - kv  50:                                split.count u16              = 7
llama_model_loader: - kv  51:                        split.tensors.count i32              = 1147
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq3_k_r4:  116 tensors
llama_model_loader: - type iq4_ks_r4:   58 tensors
llm_load_vocab: special tokens cache size = 818
llm_load_vocab: token to piece cache size = 0.8223 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 129280
llm_load_print_meta: n_merges         = 127741
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 163840
llm_load_print_meta: n_embd           = 7168
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 128
llm_load_print_meta: n_head_kv        = 128
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 24576
llm_load_print_meta: n_embd_v_gqa     = 16384
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 18432
llm_load_print_meta: n_expert         = 256
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 0.025
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = IQ3_K_R4 - 3.4325 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 300.938 GiB (3.847 BPW) 
llm_load_print_meta: repeating layers = 299.104 GiB (3.834 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1 0528
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: LF token         = 131 'Ä'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_layer_dense_lead   = 3
llm_load_print_meta: n_lora_q             = 1536
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 2048
llm_load_print_meta: n_expert_shared      = 1
llm_load_print_meta: expert_weights_scale = 2.5
llm_load_print_meta: expert_weights_norm  = 1
llm_load_print_meta: expert_gating_func   = sigmoid
llm_load_print_meta: rope_yarn_log_mul    = 0.1000
llm_load_tensors: ggml ctx size =    1.40 MiB
Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 36486.67 MiB
llm_load_tensors:        CPU buffer size = 43905.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 43905.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 44473.21 MiB
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 13995.99 MiB
llm_load_tensors:      CUDA1 buffer size = 13730.03 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =   592.89 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   573.76 MiB
llama_new_context_with_model: KV self size  = 1166.62 MiB, c^KV (q8_0): 1166.62 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  4104.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  4176.03 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   624.05 MiB
llama_new_context_with_model: graph nodes  = 8245
llama_new_context_with_model: graph splits = 149
INFO [                    init] initializing slots | tid="132484236767232" timestamp=1749450594 n_slots=1
INFO [                    init] new slot | tid="132484236767232" timestamp=1749450594 id_slot=0 n_ctx_slot=32768
INFO [                    main] model loaded | tid="132484236767232" timestamp=1749450594
INFO [                    main] chat template | tid="132484236767232" timestamp=1749450594 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
INFO [                    main] HTTP server listening | tid="132484236767232" timestamp=1749450594 n_threads_http="111" port="10002" hostname="0.0.0.0"
INFO [            update_slots] all slots are idle | tid="132484236767232" timestamp=1749450594
INFO [      log_server_request] request | tid="132482150162432" timestamp=1749450608 remote_addr="172.17.0.3" remote_port=51312 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="132482141769728" timestamp=1749450614 remote_addr="172.17.0.3" remote_port=39990 status=200 method="GET" path="/v1/models" params={}
```

```
INFO [   launch_slot_with_task] slot is processing task | tid="132484236767232" timestamp=1749450615 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="132484236767232" timestamp=1749450615 id_slot=0 id_task=0 p0=0
INFO [            update_slots] kv cache rm [p0, end) | tid="132484236767232" timestamp=1749450650 id_slot=0 id_task=0 p0=4096
INFO [            update_slots] kv cache rm [p0, end) | tid="132484236767232" timestamp=1749450686 id_slot=0 id_task=0 p0=8192
INFO [            update_slots] kv cache rm [p0, end) | tid="132484236767232" timestamp=1749450723 id_slot=0 id_task=0 p0=12288
INFO [            update_slots] kv cache rm [p0, end) | tid="132484236767232" timestamp=1749450761 id_slot=0 id_task=0 p0=16384
INFO [           print_timings] prompt eval time     =  182575.11 ms / 19385 tokens (    9.42 ms per token,   106.18 tokens per second) | tid="132484236767232" timestamp=1749450856 id_slot=0 id_task=0 t_prompt_processing=182575.108 n_prompt_tokens_processed=19385 t_token=9.418370286303844 n_tokens_second=106.1754814900616
INFO [           print_timings] generation eval time =   59163.59 ms /   538 runs   (  109.97 ms per token,     9.09 tokens per second) | tid="132484236767232" timestamp=1749450856 id_slot=0 id_task=0 t_token_generation=59163.594 n_decoded=538 t_token=109.96950557620818 n_tokens_second=9.09342999007126
INFO [           print_timings]           total time =  241738.70 ms | tid="132484236767232" timestamp=1749450856 id_slot=0 id_task=0 t_prompt_processing=182575.108 t_token_generation=59163.594 t_total=241738.702
INFO [            update_slots] slot released | tid="132484236767232" timestamp=1749450856 id_slot=0 id_task=0 n_ctx=32768 n_past=19922 n_system_tokens=0 n_cache_tokens=19922 truncated=false
INFO [            update_slots] all slots are idle | tid="132484236767232" timestamp=1749450856
INFO [      log_server_request] request | tid="132482133377024" timestamp=1749450856 remote_addr="172.17.0.3" remote_port=39998 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="132484236767232" timestamp=1749450856
```

---

👤 **mtcl** commented the **2025-06-09** at **06:36:14**:<br>

So by using two GPUs I can get 100+ tokens/second pp on the IQ3 Quants. Impressive! I also have 32k context length and I got ctk of Q8. So overall amazing results!

---

👤 **mtcl** commented the **2025-06-09** at **06:36:14**:<br>

So by using two GPUs I can get 100+ tokens/second pp on the IQ3 Quants. Impressive!

---

👤 **ikawrakow** commented the **2025-06-09** at **06:37:13**:<br>

OK, that's great! I think you have enough free VRAM to increase the context to 65k or even 131k tokens.

---

👤 **mtcl** commented the **2025-06-09** at **06:37:46**:<br>

let me try 64K, if I can do that I will be super happy!!

---

👤 **mtcl** commented the **2025-06-09** at **06:38:57**:<br>

> OK, that's great! I think you have enough free VRAM to increase the context to 65k or even 131k tokens.

Can you please review my prompt to see if this is correctly optimized?

```bash
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-server \
    --model /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf \
    --alias ubergarm/DeepSeek-R1-0528-GGUF \
    --ctx-size 32768 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 4096 -ub 4096 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    -ot "blk\.(3)\.ffn_.*=CUDA0" \
    -ot "blk\.(5)\.ffn_.*=CUDA1" \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```

i am especially looking at ot attribute there.

---

👤 **ikawrakow** commented the **2025-06-09** at **06:45:07**:<br>

You need `--ctx-size 65536` to get 65k tokens.

You are offloading only 1 layer per GPU, that is not going to make a big difference for performance (you gain in the range of 3-5%).

Btw, I'm not familiar with the Xeon CPU you have. I would be also interested to see CPU-only performance on it. To do that, you just prepend `CUDA_VISIBLE_DEVICES=` to the command line when starting the server.

---

👤 **mtcl** commented the **2025-06-09** at **06:45:16**:<br>

ok it crashed at 64K

```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-server \
    --model /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf \
    --alias ubergarm/DeepSeek-R1-0528-GGUF \
    --ctx-size 65536 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 4096 -ub 4096 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    -ot "blk\.(3)\.ffn_.*=CUDA0" \
    -ot "blk\.(5)\.ffn_.*=CUDA1" \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```

```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
INFO [                    main] build info | tid="132104685867008" timestamp=1749451448 build=3737 commit="58f08e43"
INFO [                    main] system info | tid="132104685867008" timestamp=1749451448 n_threads=57 n_threads_batch=-1 total_threads=112 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 6 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 52 key-value pairs and 1147 tensors from /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 0528
llama_model_loader: - kv   3:                            general.version str              = 0528
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek-R1
llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
llama_model_loader: - kv   6:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   7:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   8:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   9:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  10:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  11:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  12:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  13: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  15:                          general.file_type u32              = 339
llama_model_loader: - kv  16:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  17:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  18:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  19:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  20:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  21:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  22:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  23:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  24:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  25:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  26:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  27:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  28:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  29:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  30:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  31: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  32: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  33:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  34:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  36:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  37:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  38:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  39:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  40:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  41:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  42:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  43:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  44:               general.quantization_version u32              = 2
llama_model_loader: - kv  45:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/DeepSeek-R1...
llama_model_loader: - kv  46:                   quantize.imatrix.dataset str              = ubergarm-imatrix-calibration-corpus-v...
llama_model_loader: - kv  47:             quantize.imatrix.entries_count i32              = 721
llama_model_loader: - kv  48:              quantize.imatrix.chunks_count i32              = 812
llama_model_loader: - kv  49:                                   split.no u16              = 0
llama_model_loader: - kv  50:                                split.count u16              = 7
llama_model_loader: - kv  51:                        split.tensors.count i32              = 1147
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq3_k_r4:  116 tensors
llama_model_loader: - type iq4_ks_r4:   58 tensors
llm_load_vocab: special tokens cache size = 818
llm_load_vocab: token to piece cache size = 0.8223 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 129280
llm_load_print_meta: n_merges         = 127741
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 163840
llm_load_print_meta: n_embd           = 7168
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 128
llm_load_print_meta: n_head_kv        = 128
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 24576
llm_load_print_meta: n_embd_v_gqa     = 16384
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 18432
llm_load_print_meta: n_expert         = 256
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 0.025
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = IQ3_K_R4 - 3.4325 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 300.938 GiB (3.847 BPW) 
llm_load_print_meta: repeating layers = 299.104 GiB (3.834 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1 0528
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: LF token         = 131 'Ä'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_layer_dense_lead   = 3
llm_load_print_meta: n_lora_q             = 1536
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 2048
llm_load_print_meta: n_expert_shared      = 1
llm_load_print_meta: expert_weights_scale = 2.5
llm_load_print_meta: expert_weights_norm  = 1
llm_load_print_meta: expert_gating_func   = sigmoid
llm_load_print_meta: rope_yarn_log_mul    = 0.1000
llm_load_tensors: ggml ctx size =    1.40 MiB
Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 36486.67 MiB
llm_load_tensors:        CPU buffer size = 43905.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 43905.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 44473.21 MiB
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 13995.99 MiB
llm_load_tensors:      CUDA1 buffer size = 13730.03 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 65536
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =  1185.77 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =  1147.51 MiB
llama_new_context_with_model: KV self size  = 2333.25 MiB, c^KV (q8_0): 2333.25 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 7688.02 MiB on device 0: cudaMalloc failed: out of memory
ggml_gallocr_reserve_n: failed to allocate CUDA0 buffer of size 8061468672
llama_new_context_with_model: failed to allocate compute buffers
llama_init_from_gpt_params: error: failed to create context with model '/media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf'
 ERR [              load_model] unable to load model | tid="132104685867008" timestamp=1749451473 model="/media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf"
Segmentation fault (core dumped)
```

---

👤 **ikawrakow** commented the **2025-06-09** at **06:48:26**:<br>

OK, I guess you just remove `-ot "blk\.(3)\.ffn_.*=CUDA0"`  and `-ot "blk\.(5)\.ffn_.*=CUDA0"` arguments. You will get 3-5% lower performance, but you should be able to run with 65k context.

---

👤 **mtcl** commented the **2025-06-09** at **06:48:34**:<br>

OK i changed the context size to 60K and it worked:

```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-server \
    --model /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf \
    --alias ubergarm/DeepSeek-R1-0528-GGUF \
    --ctx-size 61440 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 4096 -ub 4096 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    -ot "blk\.(3)\.ffn_.*=CUDA0" \
    -ot "blk\.(5)\.ffn_.*=CUDA1" \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```

```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
INFO [                    main] build info | tid="134334334976000" timestamp=1749451619 build=3737 commit="58f08e43"
INFO [                    main] system info | tid="134334334976000" timestamp=1749451619 n_threads=57 n_threads_batch=-1 total_threads=112 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 6 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 52 key-value pairs and 1147 tensors from /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 0528
llama_model_loader: - kv   3:                            general.version str              = 0528
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek-R1
llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
llama_model_loader: - kv   6:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   7:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   8:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   9:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  10:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  11:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  12:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  13: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  15:                          general.file_type u32              = 339
llama_model_loader: - kv  16:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  17:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  18:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  19:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  20:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  21:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  22:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  23:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  24:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  25:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  26:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  27:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  28:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  29:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  30:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  31: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  32: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  33:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  34:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  36:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  37:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  38:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  39:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  40:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  41:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  42:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  43:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  44:               general.quantization_version u32              = 2
llama_model_loader: - kv  45:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/DeepSeek-R1...
llama_model_loader: - kv  46:                   quantize.imatrix.dataset str              = ubergarm-imatrix-calibration-corpus-v...
llama_model_loader: - kv  47:             quantize.imatrix.entries_count i32              = 721
llama_model_loader: - kv  48:              quantize.imatrix.chunks_count i32              = 812
llama_model_loader: - kv  49:                                   split.no u16              = 0
llama_model_loader: - kv  50:                                split.count u16              = 7
llama_model_loader: - kv  51:                        split.tensors.count i32              = 1147
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq3_k_r4:  116 tensors
llama_model_loader: - type iq4_ks_r4:   58 tensors
llm_load_vocab: special tokens cache size = 818
llm_load_vocab: token to piece cache size = 0.8223 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 129280
llm_load_print_meta: n_merges         = 127741
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 163840
llm_load_print_meta: n_embd           = 7168
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 128
llm_load_print_meta: n_head_kv        = 128
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 24576
llm_load_print_meta: n_embd_v_gqa     = 16384
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 18432
llm_load_print_meta: n_expert         = 256
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 0.025
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = IQ3_K_R4 - 3.4325 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 300.938 GiB (3.847 BPW) 
llm_load_print_meta: repeating layers = 299.104 GiB (3.834 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1 0528
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: LF token         = 131 'Ä'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_layer_dense_lead   = 3
llm_load_print_meta: n_lora_q             = 1536
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 2048
llm_load_print_meta: n_expert_shared      = 1
llm_load_print_meta: expert_weights_scale = 2.5
llm_load_print_meta: expert_weights_norm  = 1
llm_load_print_meta: expert_gating_func   = sigmoid
llm_load_print_meta: rope_yarn_log_mul    = 0.1000
llm_load_tensors: ggml ctx size =    1.40 MiB
Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 36486.67 MiB
llm_load_tensors:        CPU buffer size = 43905.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 43905.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 44473.21 MiB
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 13995.99 MiB
llm_load_tensors:      CUDA1 buffer size = 13730.03 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 61440
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =  1111.66 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =  1075.80 MiB
llama_new_context_with_model: KV self size  = 2187.42 MiB, c^KV (q8_0): 2187.42 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  7272.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  6632.03 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =  1072.05 MiB
llama_new_context_with_model: graph nodes  = 13613
llama_new_context_with_model: graph splits = 149
INFO [                    init] initializing slots | tid="134334334976000" timestamp=1749451665 n_slots=1
INFO [                    init] new slot | tid="134334334976000" timestamp=1749451665 id_slot=0 n_ctx_slot=61440
INFO [                    main] model loaded | tid="134334334976000" timestamp=1749451665
INFO [                    main] chat template | tid="134334334976000" timestamp=1749451665 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
INFO [                    main] HTTP server listening | tid="134334334976000" timestamp=1749451665 n_threads_http="111" port="10002" hostname="0.0.0.0"
INFO [            update_slots] all slots are idle | tid="134334334976000" timestamp=1749451665


```

---

👤 **mtcl** commented the **2025-06-09** at **06:49:32**:<br>

> OK, I guess you just remove `-ot "blk\.(3)\.ffn_.*=CUDA0"` and `-ot "blk\.(5)\.ffn_.*=CUDA0"` arguments. You will get 3-5% lower performance, but you should be able to run with 65k context.

Let me try this right after I run a 16k context processing right now.

---

👤 **mtcl** commented the **2025-06-09** at **06:51:52**:<br>

OK with the 60K context eventhough the server started, it crashed when i sent 16k context as a prompt:

```
erver_request] request | tid="134332240879616" timestamp=1749451794 remote_addr="172.17.0.3" remote_port=42270 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="134332232486912" timestamp=1749451795 remote_addr="172.17.0.3" remote_port=42272 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="134332148609024" timestamp=1749451800 remote_addr="172.17.0.3" remote_port=42282 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="134334334976000" timestamp=1749451801 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="134334334976000" timestamp=1749451801 id_slot=0 id_task=0 p0=0
CUDA error: out of memory
  current device: 0, in function alloc at /home/mukul/dev-ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:384
  cuMemCreate(&handle, reserve_size, &prop, 0)
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
Aborted (core dumped)
```

I will just try your suggestion now


> OK, I guess you just remove `-ot "blk\.(3)\.ffn_.*=CUDA0"` and `-ot "blk\.(5)\.ffn_.*=CUDA0"` arguments. You will get 3-5% lower performance, but you should be able to run with 65k context.

---

👤 **mtcl** commented the **2025-06-09** at **06:59:50**:<br>

Here are the results:

```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-server \
    --model /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf \
    --alias ubergarm/DeepSeek-R1-0528-GGUF \
    --ctx-size 61440 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 4096 -ub 4096 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```


```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
INFO [                    main] build info | tid="133677812011008" timestamp=1749451965 build=3737 commit="58f08e43"
INFO [                    main] system info | tid="133677812011008" timestamp=1749451965 n_threads=57 n_threads_batch=-1 total_threads=112 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 6 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 52 key-value pairs and 1147 tensors from /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 0528
llama_model_loader: - kv   3:                            general.version str              = 0528
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek-R1
llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
llama_model_loader: - kv   6:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   7:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   8:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   9:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  10:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  11:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  12:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  13: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  15:                          general.file_type u32              = 339
llama_model_loader: - kv  16:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  17:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  18:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  19:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  20:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  21:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  22:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  23:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  24:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  25:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  26:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  27:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  28:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  29:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  30:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  31: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  32: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  33:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  34:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  36:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  37:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  38:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  39:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  40:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  41:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  42:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  43:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  44:               general.quantization_version u32              = 2
llama_model_loader: - kv  45:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/DeepSeek-R1...
llama_model_loader: - kv  46:                   quantize.imatrix.dataset str              = ubergarm-imatrix-calibration-corpus-v...
llama_model_loader: - kv  47:             quantize.imatrix.entries_count i32              = 721
llama_model_loader: - kv  48:              quantize.imatrix.chunks_count i32              = 812
llama_model_loader: - kv  49:                                   split.no u16              = 0
llama_model_loader: - kv  50:                                split.count u16              = 7
llama_model_loader: - kv  51:                        split.tensors.count i32              = 1147
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq3_k_r4:  116 tensors
llama_model_loader: - type iq4_ks_r4:   58 tensors
llm_load_vocab: special tokens cache size = 818
llm_load_vocab: token to piece cache size = 0.8223 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 129280
llm_load_print_meta: n_merges         = 127741
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 163840
llm_load_print_meta: n_embd           = 7168
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 128
llm_load_print_meta: n_head_kv        = 128
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 24576
llm_load_print_meta: n_embd_v_gqa     = 16384
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 18432
llm_load_print_meta: n_expert         = 256
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 0.025
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = IQ3_K_R4 - 3.4325 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 300.938 GiB (3.847 BPW) 
llm_load_print_meta: repeating layers = 299.104 GiB (3.834 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1 0528
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: LF token         = 131 'Ä'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_layer_dense_lead   = 3
llm_load_print_meta: n_lora_q             = 1536
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 2048
llm_load_print_meta: n_expert_shared      = 1
llm_load_print_meta: expert_weights_scale = 2.5
llm_load_print_meta: expert_weights_norm  = 1
llm_load_print_meta: expert_gating_func   = sigmoid
llm_load_print_meta: rope_yarn_log_mul    = 0.1000
llm_load_tensors: ggml ctx size =    1.40 MiB
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 41735.95 MiB
llm_load_tensors:        CPU buffer size = 43905.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 43905.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 44473.21 MiB
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size =  9056.64 MiB
llm_load_tensors:      CUDA1 buffer size =  8687.38 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 61440
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =  1111.66 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =  1075.80 MiB
llama_new_context_with_model: KV self size  = 2187.42 MiB, c^KV (q8_0): 2187.42 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  7272.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  6632.03 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =  1072.05 MiB
llama_new_context_with_model: graph nodes  = 13613
llama_new_context_with_model: graph splits = 149
INFO [                    init] initializing slots | tid="133677812011008" timestamp=1749452022 n_slots=1
INFO [                    init] new slot | tid="133677812011008" timestamp=1749452022 id_slot=0 n_ctx_slot=61440
INFO [                    main] model loaded | tid="133677812011008" timestamp=1749452022
INFO [                    main] chat template | tid="133677812011008" timestamp=1749452022 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
INFO [                    main] HTTP server listening | tid="133677812011008" timestamp=1749452022 n_threads_http="111" port="10002" hostname="0.0.0.0"
```


```
INFO [            update_slots] all slots are idle | tid="133677812011008" timestamp=1749452022
INFO [      log_server_request] request | tid="133675714863104" timestamp=1749452032 remote_addr="172.17.0.3" remote_port=52820 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="133675706470400" timestamp=1749452066 remote_addr="172.17.0.3" remote_port=60306 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="133677812011008" timestamp=1749452066 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="133677812011008" timestamp=1749452066 id_slot=0 id_task=0 p0=0
INFO [            update_slots] kv cache rm [p0, end) | tid="133677812011008" timestamp=1749452106 id_slot=0 id_task=0 p0=4096
INFO [            update_slots] kv cache rm [p0, end) | tid="133677812011008" timestamp=1749452143 id_slot=0 id_task=0 p0=8192
INFO [            update_slots] kv cache rm [p0, end) | tid="133677812011008" timestamp=1749452181 id_slot=0 id_task=0 p0=12288
INFO [            update_slots] kv cache rm [p0, end) | tid="133677812011008" timestamp=1749452220 id_slot=0 id_task=0 p0=16384
INFO [           print_timings] prompt eval time     =  188356.81 ms / 17617 tokens (   10.69 ms per token,    93.53 tokens per second) | tid="133677812011008" timestamp=1749452317 id_slot=0 id_task=0 t_prompt_processing=188356.814 n_prompt_tokens_processed=17617 t_token=10.691764432082648 n_tokens_second=93.52993197262298
INFO [           print_timings] generation eval time =   62522.24 ms /   540 runs   (  115.78 ms per token,     8.64 tokens per second) | tid="133677812011008" timestamp=1749452317 id_slot=0 id_task=0 t_token_generation=62522.242 n_decoded=540 t_token=115.78192962962963 n_tokens_second=8.636926359742507
INFO [           print_timings]           total time =  250879.06 ms | tid="133677812011008" timestamp=1749452317 id_slot=0 id_task=0 t_prompt_processing=188356.814 t_token_generation=62522.242 t_total=250879.056
INFO [            update_slots] slot released | tid="133677812011008" timestamp=1749452317 id_slot=0 id_task=0 n_ctx=61440 n_past=18156 n_system_tokens=0 n_cache_tokens=18156 truncated=false
INFO [            update_slots] all slots are idle | tid="133677812011008" timestamp=1749452317
INFO [      log_server_request] request | tid="133675622592512" timestamp=1749452317 remote_addr="172.17.0.3" remote_port=60314 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="133677812011008" timestamp=1749452317

```

---

👤 **mtcl** commented the **2025-06-09** at **07:01:50**:<br>

You are absolutely the best! I can now fit 60K context with 93 tk/s prefill and 8.63 tk/s generation. So i have to pick and choose what I want the most. That helps. Thank you again!

---

👤 **ikawrakow** commented the **2025-06-09** at **07:05:30**:<br>

@saood06 Can you point me to the specific place whee the check is being made. But apart from this, I still think there is a bug because it does not make sense that the scheduler wants to allocate such insane amount of memory. I haven't come around to look why that happens.

---

👤 **mtcl** commented the **2025-06-09** at **07:09:03**:<br>

would you know if I want to do to optimize my Qwen3 startup command here? what should I change here?

this is what I had on @ubergarm 's guide

```bash
CUDA_VISIBLE_DEVICES="1" ./build/bin/llama-server \
  --model /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf \
  --alias ubergarm/Qwen3-235B-A22B-mix-IQ3_K \
  -fa \
  -ctk q8_0 -ctv q8_0 \
  -c 32768 \
  -fmoe \
  -b 1024 -ub 1024 \
  -amb 512 \
  -rtr \
  -ot blk\.1[2-9]\.ffn.*=CPU \
  -ot blk\.[2-8][0-9]\.ffn.*=CPU \
  -ot blk\.9[0-3]\.ffn.*=CPU \
  -ngl 99 \
  --threads 57 \
  --host 0.0.0.0 \
  --port 10002
```

and I am trying this now, is there something else i should bring over from above command though?

```bash
CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-server \
    --model /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf \
  --alias ubergarm/Qwen3-235B-A22B-mix-IQ3_K \
    --ctx-size 65536 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 4096 -ub 4096 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```

---

👤 **mtcl** commented the **2025-06-09** at **07:10:22**:<br>

```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-server \
    --model /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf \
  --alias ubergarm/Qwen3-235B-A22B-mix-IQ3_K \
    --ctx-size 65536 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 4096 -ub 4096 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```
```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
INFO [                    main] build info | tid="133181503258624" timestamp=1749452693 build=3737 commit="58f08e43"
INFO [                    main] system info | tid="133181503258624" timestamp=1749452693 n_threads=57 n_threads_batch=-1 total_threads=112 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 2 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 40 key-value pairs and 1131 tensors from /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   9:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv  10:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  11:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  12:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  13:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  14:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  15:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  16:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  17:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  18:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  19:                          general.file_type u32              = 139
llama_model_loader: - kv  20:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  21:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  22:               general.quantization_version u32              = 2
llama_model_loader: - kv  23:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  24:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  25:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  26:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  27:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  28:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  29:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  30:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  31:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  32:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  33:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/Qwen3-235B-...
llama_model_loader: - kv  34:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  35:             quantize.imatrix.entries_count i32              = 753
llama_model_loader: - kv  36:              quantize.imatrix.chunks_count i32              = 225
llama_model_loader: - kv  37:                                   split.no u16              = 0
llama_model_loader: - kv  38:                                split.count u16              = 3
llama_model_loader: - kv  39:                        split.tensors.count i32              = 1131
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q8_0:    2 tensors
llama_model_loader: - type iq3_k:  188 tensors
llama_model_loader: - type iq4_k:   94 tensors
llama_model_loader: - type iq6_k:  376 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 94
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 16
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 12288
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 40960
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = IQ3_K - 3.4325 bpw
llm_load_print_meta: model params     = 235.094 B
llm_load_print_meta: model size       = 106.830 GiB (3.903 BPW) 
llm_load_print_meta: repeating layers = 105.598 GiB (3.879 BPW, 233.849 B parameters)
llm_load_print_meta: general.name     = Qwen3 235B A22B
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    1.49 MiB
Tensor blk.0.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.0.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.0.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.1.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.1.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.1.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 63 repeating layers to GPU
llm_load_tensors: offloaded 63/95 layers to GPU
llm_load_tensors:        CPU buffer size = 36422.69 MiB
llm_load_tensors:        CPU buffer size = 37141.03 MiB
llm_load_tensors:        CPU buffer size = 35082.59 MiB
llm_load_tensors:        CPU buffer size = 36291.28 MiB
llm_load_tensors:        CPU buffer size =  1722.64 MiB
llm_load_tensors:      CUDA0 buffer size =  1808.69 MiB
llm_load_tensors:      CUDA1 buffer size =  1867.03 MiB
....................................................................................................
=====================================================================
 MLA is only available for LLM_ARCH_DEEPSEEK2 -> turning off MLA
=====================================================================
llama_new_context_with_model: n_ctx      = 65536
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
ggml_cuda_host_malloc: failed to allocate 3038.00 MiB of pinned memory: invalid argument
llama_kv_cache_init:        CPU KV buffer size =  3038.00 MiB
llama_kv_cache_init:      CUDA0 KV buffer size =  3038.02 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =  3136.02 MiB
llama_new_context_with_model: KV self size  = 9212.00 MiB, K (q8_0): 3196.00 MiB, V (f16): 6016.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.16 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =  3068.61 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =   896.03 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =  1088.05 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 595
INFO [                    init] initializing slots | tid="133181503258624" timestamp=1749452713 n_slots=1
INFO [                    init] new slot | tid="133181503258624" timestamp=1749452713 id_slot=0 n_ctx_slot=65536
INFO [                    main] model loaded | tid="133181503258624" timestamp=1749452713
INFO [                    main] chat template | tid="133181503258624" timestamp=1749452713 chat_example="<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n" built_in=true
INFO [                    main] HTTP server listening | tid="133181503258624" timestamp=1749452713 n_threads_http="111" port="10002" hostname="0.0.0.0"
INFO [            update_slots] all slots are idle | tid="133181503258624" timestamp=1749452713
INFO [      log_server_request] request | tid="133179400769536" timestamp=1749452792 remote_addr="172.17.0.3" 
INFO [   launch_slot_with_task] slot is processing task | tid="133181503258624" timestamp=1749452800 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="133181503258624" timestamp=1749452800 id_slot=0 id_task=0 p0=0
INFO [            update_slots] kv cache rm [p0, end) | tid="133181503258624" timestamp=1749452820 id_slot=0 id_task=0 p0=4096
INFO [            update_slots] kv cache rm [p0, end) | tid="133181503258624" timestamp=1749452834 id_slot=0 id_task=0 p0=8192
INFO [            update_slots] kv cache rm [p0, end) | tid="133181503258624" timestamp=1749452848 id_slot=0 id_task=0 p0=12288
INFO [            update_slots] kv cache rm [p0, end) | tid="133181503258624" timestamp=1749452862 id_slot=0 id_task=0 p0=16384
INFO [            update_slots] kv cache rm [p0, end) | tid="133181503258624" timestamp=1749452877 id_slot=0 id_task=0 p0=20480
.INFO [           print_timings] prompt eval time     =   89767.54 ms / 21880 tokens (    4.10 ms per token,   243.74 tokens per second) | tid="133181503258624" timestamp=1749452963 id_slot=0 id_task=0 t_prompt_processing=89767.542 n_prompt_tokens_processed=21880 t_token=4.1027212979890315 n_tokens_second=243.74066073904527
INFO [           print_timings] generation eval time =   72821.50 ms /   563 runs   (  129.35 ms per token,     7.73 tokens per second) | tid="133181503258624" timestamp=1749452963 id_slot=0 id_task=0 t_token_generation=72821.5 n_decoded=563 t_token=129.34547069271758 n_tokens_second=7.731233220958097
INFO [           print_timings]           total time =  162589.04 ms | tid="133181503258624" timestamp=1749452963 id_slot=0 id_task=0 t_prompt_processing=89767.542 t_token_generation=72821.5 t_total=162589.04200000002
INFO [            update_slots] slot released | tid="133181503258624" timestamp=1749452963 id_slot=0 id_task=0 n_ctx=65536 n_past=22442 n_system_tokens=0 n_cache_tokens=22442 truncated=false
INFO [            update_slots] all slots are idle | tid="133181503258624" timestamp=1749452963
INFO [      log_server_request] request | tid="133179285434368" timestamp=1749452963 remote_addr="172.17.0.3" remote_port=47690 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="133181503258624" timestamp=1749452963
```

---

👤 **ikawrakow** commented the **2025-06-09** at **07:17:58**:<br>

* You need `-ctv q8_0` to also have the V cache quantized.
* You need to change `--n-gpu-layers` to 100 (Qwen3 has more layers than DeepSeek
* Remove `-mla` (not applicable to any model other than DeepSeek)
* You don't need the `-amb` argument

Let's see what buffer sizes you get with that. Then we will know how many layers you can put on the GPU.

---

👤 **ikawrakow** commented the **2025-06-09** at **07:19:10**:<br>

Oh, try changing threads to 56 from 57. 57 is a really strange number of threads.

---

👤 **saood06** commented the **2025-06-09** at **07:19:50**:<br>

> [@saood06](https://github.com/saood06) Can you point me to the specific place whee the check is being made. 

@ikawrakow 

https://github.com/ikawrakow/ik_llama.cpp/blob/58f08e43859a942dcc4d585f04b729eb50603264/src/llama.cpp#L20758

>But apart from this, I still think there is a bug because it does not make sense that the scheduler wants to allocate such insane amount of memory.

Yes that doesn't make much sense to me either.

> I haven't come around to look why that happens.

If you ever do I'd be interested to hear anything you find out.

---

👤 **saood06** commented the **2025-06-09** at **07:19:50**:<br>

> [@saood06](https://github.com/saood06) Can you point me to the specific place whee the check is being made. 

https://github.com/ikawrakow/ik_llama.cpp/blob/58f08e43859a942dcc4d585f04b729eb50603264/src/llama.cpp#L20758

>But apart from this, I still think there is a bug because it does not make sense that the scheduler wants to allocate such insane amount of memory.

Yes that doesn't make much sense to me either.

> I haven't come around to look why that happens.

If you ever do I'd be interested to hear anything you find out.

---

👤 **mtcl** commented the **2025-06-09** at **07:20:00**:<br>

ok i tried something before your comment, so I will post that here anyway, and then I will now try your settings in there. 

below is from what I was experimenting with 32k

```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0,1" ./build/bin/llama-server \
  --model /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf \
  --alias ubergarm/Qwen3-235B-A22B-mix-IQ3_K \
  -fa \
  -ctk q8_0 -ctv q8_0 \
  -c 32768 \
  -fmoe \
  -b 4096 -ub 4096 \
  -amb 512 \
  -rtr \
  -ot blk\.1[2-9]\.ffn.*=CPU \
  -ot blk\.[2-8][0-9]\.ffn.*=CPU \
  -ot blk\.9[0-3]\.ffn.*=CPU \
  -ngl 99 \
  --threads 57 \
  --host 0.0.0.0 \
  --port 10002
```

```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
INFO [                    main] build info | tid="135402454921216" timestamp=1749453237 build=3737 commit="58f08e43"
INFO [                    main] system info | tid="135402454921216" timestamp=1749453237 n_threads=57 n_threads_batch=-1 total_threads=112 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 2 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 40 key-value pairs and 1131 tensors from /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   9:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv  10:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  11:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  12:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  13:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  14:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  15:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  16:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  17:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  18:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  19:                          general.file_type u32              = 139
llama_model_loader: - kv  20:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  21:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  22:               general.quantization_version u32              = 2
llama_model_loader: - kv  23:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  24:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  25:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  26:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  27:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  28:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  29:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  30:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  31:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  32:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  33:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/Qwen3-235B-...
llama_model_loader: - kv  34:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  35:             quantize.imatrix.entries_count i32              = 753
llama_model_loader: - kv  36:              quantize.imatrix.chunks_count i32              = 225
llama_model_loader: - kv  37:                                   split.no u16              = 0
llama_model_loader: - kv  38:                                split.count u16              = 3
llama_model_loader: - kv  39:                        split.tensors.count i32              = 1131
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q8_0:    2 tensors
llama_model_loader: - type iq3_k:  188 tensors
llama_model_loader: - type iq4_k:   94 tensors
llama_model_loader: - type iq6_k:  376 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 94
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 16
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 12288
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 40960
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = IQ3_K - 3.4325 bpw
llm_load_print_meta: model params     = 235.094 B
llm_load_print_meta: model size       = 106.830 GiB (3.903 BPW) 
llm_load_print_meta: repeating layers = 105.598 GiB (3.879 BPW, 233.849 B parameters)
llm_load_print_meta: general.name     = Qwen3 235B A22B
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    1.49 MiB
Tensor blk.12.ffn_norm.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_norm.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_norm.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_norm.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_norm.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_norm.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_norm.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_norm.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_norm.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_norm.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_norm.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_norm.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_norm.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_norm.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_norm.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_norm.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_norm.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_norm.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_norm.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_norm.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_norm.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_norm.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_norm.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_norm.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_norm.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_norm.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_norm.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_norm.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_norm.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_norm.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_norm.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_norm.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_norm.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_norm.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_norm.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_norm.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_norm.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_norm.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_norm.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_norm.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_norm.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_norm.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_norm.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_norm.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_norm.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_norm.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_norm.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_norm.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_norm.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_norm.weight buffer type overriden to CPU
Tensor blk.61.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.61.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_norm.weight buffer type overriden to CPU
Tensor blk.62.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.62.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_norm.weight buffer type overriden to CPU
Tensor blk.63.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.63.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_norm.weight buffer type overriden to CPU
Tensor blk.64.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.64.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_norm.weight buffer type overriden to CPU
Tensor blk.65.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.65.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_norm.weight buffer type overriden to CPU
Tensor blk.66.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.66.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_norm.weight buffer type overriden to CPU
Tensor blk.67.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.67.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_norm.weight buffer type overriden to CPU
Tensor blk.68.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.68.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_norm.weight buffer type overriden to CPU
Tensor blk.69.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.69.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_norm.weight buffer type overriden to CPU
Tensor blk.70.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.70.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_norm.weight buffer type overriden to CPU
Tensor blk.71.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.71.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_norm.weight buffer type overriden to CPU
Tensor blk.72.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.72.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_norm.weight buffer type overriden to CPU
Tensor blk.73.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.73.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_norm.weight buffer type overriden to CPU
Tensor blk.74.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.74.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_norm.weight buffer type overriden to CPU
Tensor blk.75.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.75.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_norm.weight buffer type overriden to CPU
Tensor blk.76.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.76.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_norm.weight buffer type overriden to CPU
Tensor blk.77.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.77.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_norm.weight buffer type overriden to CPU
Tensor blk.78.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.78.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_norm.weight buffer type overriden to CPU
Tensor blk.79.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.79.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_norm.weight buffer type overriden to CPU
Tensor blk.80.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.80.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_norm.weight buffer type overriden to CPU
Tensor blk.81.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.81.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_norm.weight buffer type overriden to CPU
Tensor blk.82.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.82.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_norm.weight buffer type overriden to CPU
Tensor blk.83.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.83.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_norm.weight buffer type overriden to CPU
Tensor blk.84.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.84.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_norm.weight buffer type overriden to CPU
Tensor blk.85.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.85.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_norm.weight buffer type overriden to CPU
Tensor blk.86.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.86.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_norm.weight buffer type overriden to CPU
Tensor blk.87.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.87.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_norm.weight buffer type overriden to CPU
Tensor blk.88.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.88.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_norm.weight buffer type overriden to CPU
Tensor blk.89.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.89.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_norm.weight buffer type overriden to CPU
Tensor blk.90.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.90.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_norm.weight buffer type overriden to CPU
Tensor blk.91.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.91.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_norm.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_norm.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:        CPU buffer size = 89709.28 MiB
llm_load_tensors:  CUDA_Host buffer size =   630.59 MiB
llm_load_tensors:      CUDA0 buffer size = 15775.66 MiB
llm_load_tensors:      CUDA1 buffer size =  3278.08 MiB
....................................................................................................
============ Repacked 246 tensors
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  1598.02 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =  1598.02 MiB
llama_new_context_with_model: KV self size  = 3196.00 MiB, K (q8_0): 1598.00 MiB, V (q8_0): 1598.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.16 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  2096.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  2502.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   576.05 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 378
INFO [                    init] initializing slots | tid="135402454921216" timestamp=1749453354 n_slots=1
INFO [                    init] new slot | tid="135402454921216" timestamp=1749453354 id_slot=0 n_ctx_slot=32768
INFO [                    main] model loaded | tid="135402454921216" timestamp=1749453354
INFO [                    main] chat template | tid="135402454921216" timestamp=1749453354 chat_example="<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n" built_in=true
INFO [                    main] HTTP server listening | tid="135402454921216" timestamp=1749453354 n_threads_http="111" port="10002" hostname="0.0.0.0"
INFO [            update_slots] all slots are idle | tid="135402454921216" timestamp=1749453354
INFO [      log_server_request] request | tid="135400345559040" timestamp=1749453357 remote_addr="172.17.0.3" remote_port=37824 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="135400337166336" timestamp=1749453360 remote_addr="172.17.0.3" remote_port=37832 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="135402454921216" timestamp=1749453362 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="135402454921216" timestamp=1749453362 id_slot=0 id_task=0 p0=0
INFO [            update_slots] kv cache rm [p0, end) | tid="135402454921216" timestamp=1749453376 id_slot=0 id_task=0 p0=4096
INFO [            update_slots] kv cache rm [p0, end) | tid="135402454921216" timestamp=1749453389 id_slot=0 id_task=0 p0=8192
INFO [            update_slots] kv cache rm [p0, end) | tid="135402454921216" timestamp=1749453403 id_slot=0 id_task=0 p0=12288
INFO [            update_slots] kv cache rm [p0, end) | tid="135402454921216" timestamp=1749453418 id_slot=0 id_task=0 p0=16384
INFO [            update_slots] kv cache rm [p0, end) | tid="135402454921216" timestamp=1749453433 id_slot=0 id_task=0 p0=20480
INFO [           print_timings] prompt eval time     =   82402.70 ms / 21880 tokens (    3.77 ms per token,   265.53 tokens per second) | tid="135402454921216" timestamp=1749453488 id_slot=0 id_task=0 t_prompt_processing=82402.695 n_prompt_tokens_processed=21880 t_token=3.7661195155393057 n_tokens_second=265.52529622969246
INFO [           print_timings] generation eval time =   43959.36 ms /   547 runs   (   80.36 ms per token,    12.44 tokens per second) | tid="135402454921216" timestamp=1749453488 id_slot=0 id_task=0 t_token_generation=43959.358 n_decoded=547 t_token=80.36445703839122 n_tokens_second=12.443311842725272
INFO [           print_timings]           total time =  126362.05 ms | tid="135402454921216" timestamp=1749453488 id_slot=0 id_task=0 t_prompt_processing=82402.695 t_token_generation=43959.358 t_total=126362.05300000001
INFO [            update_slots] slot released | tid="135402454921216" timestamp=1749453488 id_slot=0 id_task=0 n_ctx=32768 n_past=22426 n_system_tokens=0 n_cache_tokens=22426 truncated=false
INFO [            update_slots] all slots are idle | tid="135402454921216" timestamp=1749453488
INFO [      log_server_request] request | tid="135400328773632" timestamp=1749453488 remote_addr="172.17.0.3" remote_port=48428 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="135402454921216" timestamp=1749453488

```

---

👤 **mtcl** commented the **2025-06-09** at **07:26:45**:<br>

> * You need `-ctv q8_0` to also have the V cache quantized.
> 
>     * You need to change `--n-gpu-layers` to 100 (Qwen3 has more layers than DeepSeek
> 
>     * Remove `-mla` (not applicable to any model other than DeepSeek)
> 
>     * You don't need the `-amb` argument
> 
> 
> Let's see what buffer sizes you get with that. Then we will know how many layers you can put on the GPU.

Here are the results (both of my GPUs have about 10GB VRAM occupied now)

```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-server \
    --model /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf \
  --alias ubergarm/Qwen3-235B-A22B-mix-IQ3_K \
    --ctx-size 65536 \
    -ctk q8_0 -ctv q8_0 \
    -fa \
    -b 4096 -ub 4096 \
    -fmoe \
    --n-gpu-layers 100 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 56 \
    --host 0.0.0.0 \
    --port 10002
```

```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
INFO [                    main] build info | tid="126884355235840" timestamp=1749453754 build=3737 commit="58f08e43"
INFO [                    main] system info | tid="126884355235840" timestamp=1749453754 n_threads=56 n_threads_batch=-1 total_threads=112 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 2 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 40 key-value pairs and 1131 tensors from /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   9:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv  10:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  11:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  12:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  13:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  14:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  15:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  16:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  17:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  18:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  19:                          general.file_type u32              = 139
llama_model_loader: - kv  20:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  21:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  22:               general.quantization_version u32              = 2
llama_model_loader: - kv  23:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  24:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  25:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  26:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  27:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  28:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  29:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  30:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  31:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  32:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  33:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/Qwen3-235B-...
llama_model_loader: - kv  34:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  35:             quantize.imatrix.entries_count i32              = 753
llama_model_loader: - kv  36:              quantize.imatrix.chunks_count i32              = 225
llama_model_loader: - kv  37:                                   split.no u16              = 0
llama_model_loader: - kv  38:                                split.count u16              = 3
llama_model_loader: - kv  39:                        split.tensors.count i32              = 1131
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q8_0:    2 tensors
llama_model_loader: - type iq3_k:  188 tensors
llama_model_loader: - type iq4_k:   94 tensors
llama_model_loader: - type iq6_k:  376 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 94
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 16
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 12288
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 40960
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = IQ3_K - 3.4325 bpw
llm_load_print_meta: model params     = 235.094 B
llm_load_print_meta: model size       = 106.830 GiB (3.903 BPW) 
llm_load_print_meta: repeating layers = 105.598 GiB (3.879 BPW, 233.849 B parameters)
llm_load_print_meta: general.name     = Qwen3 235B A22B
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    1.49 MiB
Tensor blk.0.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.0.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.0.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.1.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.1.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.1.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:        CPU buffer size = 36422.69 MiB
llm_load_tensors:        CPU buffer size = 37141.03 MiB
llm_load_tensors:        CPU buffer size = 35082.59 MiB
llm_load_tensors:        CPU buffer size =   630.59 MiB
llm_load_tensors:      CUDA0 buffer size =  2742.20 MiB
llm_load_tensors:      CUDA1 buffer size =  3372.81 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 65536
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  3196.02 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =  3196.02 MiB
llama_new_context_with_model: KV self size  = 6392.00 MiB, K (q8_0): 3196.00 MiB, V (q8_0): 3196.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.16 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  2432.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  2502.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =  1088.05 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 238
INFO [                    init] initializing slots | tid="126884355235840" timestamp=1749453800 n_slots=1
INFO [                    init] new slot | tid="126884355235840" timestamp=1749453800 id_slot=0 n_ctx_slot=65536
INFO [                    main] model loaded | tid="126884355235840" timestamp=1749453800
INFO [                    main] chat template | tid="126884355235840" timestamp=1749453800 chat_example="<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n" built_in=true
INFO [                    main] HTTP server listening | tid="126884355235840" timestamp=1749453800 n_threads_http="111" port="10002" hostname="0.0.0.0"
INFO [            update_slots] all slots are idle | tid="126884355235840" timestamp=1749453800
INFO [      log_server_request] request | tid="126882284560384" timestamp=1749453803 remote_addr="172.17.0.3" remote_port=55816 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="126882276167680" timestamp=1749453805 remote_addr="172.17.0.3" remote_port=55832 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="126884355235840" timestamp=1749453805 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="126884355235840" timestamp=1749453805 id_slot=0 id_task=0 p0=0
INFO [            update_slots] kv cache rm [p0, end) | tid="126884355235840" timestamp=1749453821 id_slot=0 id_task=0 p0=4096
INFO [            update_slots] kv cache rm [p0, end) | tid="126884355235840" timestamp=1749453835 id_slot=0 id_task=0 p0=8192
INFO [            update_slots] kv cache rm [p0, end) | tid="126884355235840" timestamp=1749453849 id_slot=0 id_task=0 p0=12288
INFO [            update_slots] kv cache rm [p0, end) | tid="126884355235840" timestamp=1749453863 id_slot=0 id_task=0 p0=16384
INFO [            update_slots] kv cache rm [p0, end) | tid="126884355235840" timestamp=1749453877 id_slot=0 id_task=0 p0=20480
INFO [      log_server_request] request | tid="126882183897088" timestamp=1749453927 remote_addr="172.17.0.3" remote_port=34580 status=200 method="GET" path="/v1/models" params={}
INFO [           print_timings] prompt eval time     =   83617.03 ms / 21880 tokens (    3.82 ms per token,   261.67 tokens per second) | tid="126884355235840" timestamp=1749453936 id_slot=0 id_task=0 t_prompt_processing=83617.034 n_prompt_tokens_processed=21880 t_token=3.821619469835466 n_tokens_second=261.6691713796019
INFO [           print_timings] generation eval time =   46598.42 ms /   473 runs   (   98.52 ms per token,    10.15 tokens per second) | tid="126884355235840" timestamp=1749453936 id_slot=0 id_task=0 t_token_generation=46598.424 n_decoded=473 t_token=98.51675264270612 n_tokens_second=10.150557881528353
INFO [           print_timings]           total time =  130215.46 ms | tid="126884355235840" timestamp=1749453936 id_slot=0 id_task=0 t_prompt_processing=83617.034 t_token_generation=46598.424 t_total=130215.458
INFO [            update_slots] slot released | tid="126884355235840" timestamp=1749453936 id_slot=0 id_task=0 n_ctx=65536 n_past=22352 n_system_tokens=0 n_cache_tokens=22352 truncated=false
INFO [            update_slots] all slots are idle | tid="126884355235840" timestamp=1749453936
INFO [      log_server_request] request | tid="126882192289792" timestamp=1749453936 remote_addr="172.17.0.3" remote_port=55846 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="126884355235840" timestamp=1749453936
```

---

👤 **mtcl** commented the **2025-06-09** at **07:26:45**:<br>

> * You need `-ctv q8_0` to also have the V cache quantized.
> 
>     * You need to change `--n-gpu-layers` to 100 (Qwen3 has more layers than DeepSeek
> 
>     * Remove `-mla` (not applicable to any model other than DeepSeek)
> 
>     * You don't need the `-amb` argument
> 
> 
> Let's see what buffer sizes you get with that. Then we will know how many layers you can put on the GPU.

Here are the results:

```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-server \
    --model /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf \
  --alias ubergarm/Qwen3-235B-A22B-mix-IQ3_K \
    --ctx-size 65536 \
    -ctk q8_0 -ctv q8_0 \
    -fa \
    -b 4096 -ub 4096 \
    -fmoe \
    --n-gpu-layers 100 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 56 \
    --host 0.0.0.0 \
    --port 10002
```

```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
INFO [                    main] build info | tid="126884355235840" timestamp=1749453754 build=3737 commit="58f08e43"
INFO [                    main] system info | tid="126884355235840" timestamp=1749453754 n_threads=56 n_threads_batch=-1 total_threads=112 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 2 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 40 key-value pairs and 1131 tensors from /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   9:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv  10:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  11:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  12:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  13:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  14:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  15:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  16:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  17:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  18:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  19:                          general.file_type u32              = 139
llama_model_loader: - kv  20:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  21:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  22:               general.quantization_version u32              = 2
llama_model_loader: - kv  23:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  24:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  25:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  26:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  27:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  28:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  29:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  30:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  31:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  32:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  33:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/Qwen3-235B-...
llama_model_loader: - kv  34:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  35:             quantize.imatrix.entries_count i32              = 753
llama_model_loader: - kv  36:              quantize.imatrix.chunks_count i32              = 225
llama_model_loader: - kv  37:                                   split.no u16              = 0
llama_model_loader: - kv  38:                                split.count u16              = 3
llama_model_loader: - kv  39:                        split.tensors.count i32              = 1131
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q8_0:    2 tensors
llama_model_loader: - type iq3_k:  188 tensors
llama_model_loader: - type iq4_k:   94 tensors
llama_model_loader: - type iq6_k:  376 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 94
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 16
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 12288
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 40960
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = IQ3_K - 3.4325 bpw
llm_load_print_meta: model params     = 235.094 B
llm_load_print_meta: model size       = 106.830 GiB (3.903 BPW) 
llm_load_print_meta: repeating layers = 105.598 GiB (3.879 BPW, 233.849 B parameters)
llm_load_print_meta: general.name     = Qwen3 235B A22B
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    1.49 MiB
Tensor blk.0.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.0.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.0.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.1.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.1.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.1.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:        CPU buffer size = 36422.69 MiB
llm_load_tensors:        CPU buffer size = 37141.03 MiB
llm_load_tensors:        CPU buffer size = 35082.59 MiB
llm_load_tensors:        CPU buffer size =   630.59 MiB
llm_load_tensors:      CUDA0 buffer size =  2742.20 MiB
llm_load_tensors:      CUDA1 buffer size =  3372.81 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 65536
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  3196.02 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =  3196.02 MiB
llama_new_context_with_model: KV self size  = 6392.00 MiB, K (q8_0): 3196.00 MiB, V (q8_0): 3196.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.16 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  2432.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  2502.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =  1088.05 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 238
INFO [                    init] initializing slots | tid="126884355235840" timestamp=1749453800 n_slots=1
INFO [                    init] new slot | tid="126884355235840" timestamp=1749453800 id_slot=0 n_ctx_slot=65536
INFO [                    main] model loaded | tid="126884355235840" timestamp=1749453800
INFO [                    main] chat template | tid="126884355235840" timestamp=1749453800 chat_example="<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n" built_in=true
INFO [                    main] HTTP server listening | tid="126884355235840" timestamp=1749453800 n_threads_http="111" port="10002" hostname="0.0.0.0"
INFO [            update_slots] all slots are idle | tid="126884355235840" timestamp=1749453800
INFO [      log_server_request] request | tid="126882284560384" timestamp=1749453803 remote_addr="172.17.0.3" remote_port=55816 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="126882276167680" timestamp=1749453805 remote_addr="172.17.0.3" remote_port=55832 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="126884355235840" timestamp=1749453805 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="126884355235840" timestamp=1749453805 id_slot=0 id_task=0 p0=0
INFO [            update_slots] kv cache rm [p0, end) | tid="126884355235840" timestamp=1749453821 id_slot=0 id_task=0 p0=4096
INFO [            update_slots] kv cache rm [p0, end) | tid="126884355235840" timestamp=1749453835 id_slot=0 id_task=0 p0=8192
INFO [            update_slots] kv cache rm [p0, end) | tid="126884355235840" timestamp=1749453849 id_slot=0 id_task=0 p0=12288
INFO [            update_slots] kv cache rm [p0, end) | tid="126884355235840" timestamp=1749453863 id_slot=0 id_task=0 p0=16384
INFO [            update_slots] kv cache rm [p0, end) | tid="126884355235840" timestamp=1749453877 id_slot=0 id_task=0 p0=20480
INFO [      log_server_request] request | tid="126882183897088" timestamp=1749453927 remote_addr="172.17.0.3" remote_port=34580 status=200 method="GET" path="/v1/models" params={}
INFO [           print_timings] prompt eval time     =   83617.03 ms / 21880 tokens (    3.82 ms per token,   261.67 tokens per second) | tid="126884355235840" timestamp=1749453936 id_slot=0 id_task=0 t_prompt_processing=83617.034 n_prompt_tokens_processed=21880 t_token=3.821619469835466 n_tokens_second=261.6691713796019
INFO [           print_timings] generation eval time =   46598.42 ms /   473 runs   (   98.52 ms per token,    10.15 tokens per second) | tid="126884355235840" timestamp=1749453936 id_slot=0 id_task=0 t_token_generation=46598.424 n_decoded=473 t_token=98.51675264270612 n_tokens_second=10.150557881528353
INFO [           print_timings]           total time =  130215.46 ms | tid="126884355235840" timestamp=1749453936 id_slot=0 id_task=0 t_prompt_processing=83617.034 t_token_generation=46598.424 t_total=130215.458
INFO [            update_slots] slot released | tid="126884355235840" timestamp=1749453936 id_slot=0 id_task=0 n_ctx=65536 n_past=22352 n_system_tokens=0 n_cache_tokens=22352 truncated=false
INFO [            update_slots] all slots are idle | tid="126884355235840" timestamp=1749453936
INFO [      log_server_request] request | tid="126882192289792" timestamp=1749453936 remote_addr="172.17.0.3" remote_port=55846 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="126884355235840" timestamp=1749453936
```

---

👤 **ikawrakow** commented the **2025-06-09** at **07:32:50**:<br>

So, it looks like you can put about 15 layers on each GPU. Try adding before the CPU override
```
-ot "blk\.[0-9]\.ffn=CUDA0,blk\.1[0-4]\.ffn=CUDA0 \
-ot "blk\.1[5-9]\.ffn=CUDA1,blk\.2[0-9]\.ffn=CUDA1
```
If it crashes with OOM, keep reducing the number of layers by 1 until it runs (but adjust the regex so the offloaded layers are consecutive).

---

👤 **mtcl** commented the **2025-06-09** at **07:34:05**:<br>

so something like this?

```bash
CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-server \
    --model /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf \
  --alias ubergarm/Qwen3-235B-A22B-mix-IQ3_K \
    --ctx-size 65536 \
    -ctk q8_0 -ctv q8_0 \
    -fa \
    -b 4096 -ub 4096 \
    -fmoe \
    --n-gpu-layers 100 \
    -ot "blk\.[0-9]\.ffn=CUDA0,blk\.1[0-4]\.ffn=CUDA0 \
    -ot "blk\.1[5-9]\.ffn=CUDA1,blk\.2[0-9]\.ffn=CUDA1 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 56 \
    --host 0.0.0.0 \
    --port 10002
```


```
.
.
.
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:        CPU buffer size = 19167.52 MiB
llm_load_tensors:        CPU buffer size = 37141.03 MiB
llm_load_tensors:        CPU buffer size = 35082.59 MiB
llm_load_tensors:        CPU buffer size =   630.59 MiB
llm_load_tensors:      CUDA0 buffer size = 19102.05 MiB
llm_load_tensors:      CUDA1 buffer size = 14312.97 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 65536
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  3196.02 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =  3196.02 MiB
llama_new_context_with_model: KV self size  = 6392.00 MiB, K (q8_0): 3196.00 MiB, V (q8_0): 3196.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.16 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 2432.02 MiB on device 0: cudaMalloc failed: out of memory
ggml_gallocr_reserve_n: failed to allocate CUDA0 buffer of size 2550153216
llama_new_context_with_model: failed to allocate compute buffers
llama_init_from_gpt_params: error: failed to create context with model '/media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf'
 ERR [              load_model] unable to load model | tid="137730945286144" timestamp=1749454478 model="/media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf"
Segmentation fault (core dumped)
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ 
```

---

👤 **mtcl** commented the **2025-06-09** at **07:34:05**:<br>

so something like this?

```bash
CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-server \
    --model /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf \
  --alias ubergarm/Qwen3-235B-A22B-mix-IQ3_K \
    --ctx-size 65536 \
    -ctk q8_0 -ctv q8_0 \
    -fa \
    -b 4096 -ub 4096 \
    -fmoe \
    --n-gpu-layers 100 \
    -ot "blk\.[0-9]\.ffn=CUDA0,blk\.1[0-4]\.ffn=CUDA0 \
    -ot "blk\.1[5-9]\.ffn=CUDA1,blk\.2[0-9]\.ffn=CUDA1 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 56 \
    --host 0.0.0.0 \
    --port 10002
```

---

👤 **mtcl** commented the **2025-06-09** at **07:37:16**:<br>

> If it crashes with OOM, keep reducing the number of layers by 1 until it runs (but adjust the regex so the offloaded layers are consecutive).

how do i do that? below is my current command:

```
CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-server \
    --model /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf \
  --alias ubergarm/Qwen3-235B-A22B-mix-IQ3_K \
    --ctx-size 65536 \
    -ctk q8_0 -ctv q8_0 \
    -fa \
    -b 4096 -ub 4096 \
    -fmoe \
    --n-gpu-layers 100 \
    -ot "blk\.[0-9]\.ffn=CUDA0,blk\.1[0-4]\.ffn=CUDA0 \
    -ot "blk\.1[5-9]\.ffn=CUDA1,blk\.2[0-9]\.ffn=CUDA1 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 56 \
    --host 0.0.0.0 \
    --port 10002
```

---

👤 **ikawrakow** commented the **2025-06-09** at **07:41:21**:<br>

Oh, sorry, it is missing the closing quotes. It must be
```
   -ot "blk\.[0-9]\.ffn=CUDA0,blk\.1[0-4]\.ffn=CUDA0" \
   -ot "blk\.1[5-9]\.ffn=CUDA1,blk\.2[0-9]\.ffn=CUDA1" \
```

If that doesn't run, you change to
```
-ot "blk\.[0-9]\.ffn=CUDA0,blk\.1[0-3]\.ffn=CUDA0" \
-ot "blk\.1[4-9]\.ffn=CUDA1,blk\.2[0-7]\.ffn=CUDA0" \
```
etc.

---

👤 **saood06** commented the **2025-06-09** at **08:05:23**:<br>

> ```
> -ot "blk\.[0-9]\.ffn=CUDA0,blk\.1[0-3]\.ffn=CUDA0" \
> -ot "blk\.1[4-9]\.ffn=CUDA1,blk\.2[0-7]\.ffn=CUDA0" \
> ```


Just to be clear the second line should be 

> ```
> -ot "blk\.1[4-9]\.ffn=CUDA1,blk\.2[0-7]\.ffn=CUDA1" \
> ```

---

👤 **mtcl** commented the **2025-06-09** at **08:09:04**:<br>

ok, i used qwen3 to help and i started reducing one by one and this is the one that works with 22GB per GPU:

```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0,1" ./build/bin/llama-server \
  --model /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/\
Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf \
  --alias ubergarm/Qwen3-235B-A22B-mix-IQ3_K \
  --ctx-size 65536 \
  -ctk q8_0 -ctv q8_0 \
  -fa \
  -b 4096 -ub 4096 \
  -fmoe \
  --n-gpu-layers 100 \
  -ot "blk\.[0-9]\.ffn=CUDA0,blk\.1[0-1]\.ffn=CUDA0" \
  -ot "blk\.1[5-8]\.ffn=CUDA1,blk\.2[0-7]\.ffn=CUDA1" \
  --override-tensor exps=CPU \
  --parallel 1 \
  --threads 56 \
  --host 0.0.0.0 \
  --port 10002
```

```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
INFO [                    main] build info | tid="137280187613184" timestamp=1749456039 build=3737 commit="58f08e43"
INFO [                    main] system info | tid="137280187613184" timestamp=1749456039 n_threads=56 n_threads_batch=-1 total_threads=112 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 2 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 40 key-value pairs and 1131 tensors from /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   9:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv  10:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  11:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  12:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  13:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  14:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  15:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  16:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  17:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  18:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  19:                          general.file_type u32              = 139
llama_model_loader: - kv  20:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  21:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  22:               general.quantization_version u32              = 2
llama_model_loader: - kv  23:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  24:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  25:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  26:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  27:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  28:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  29:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  30:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  31:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  32:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  33:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/Qwen3-235B-...
llama_model_loader: - kv  34:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  35:             quantize.imatrix.entries_count i32              = 753
llama_model_loader: - kv  36:              quantize.imatrix.chunks_count i32              = 225
llama_model_loader: - kv  37:                                   split.no u16              = 0
llama_model_loader: - kv  38:                                split.count u16              = 3
llama_model_loader: - kv  39:                        split.tensors.count i32              = 1131
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q8_0:    2 tensors
llama_model_loader: - type iq3_k:  188 tensors
llama_model_loader: - type iq4_k:   94 tensors
llama_model_loader: - type iq6_k:  376 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 94
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 16
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 12288
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 40960
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = IQ3_K - 3.4325 bpw
llm_load_print_meta: model params     = 235.094 B
llm_load_print_meta: model size       = 106.830 GiB (3.903 BPW) 
llm_load_print_meta: repeating layers = 105.598 GiB (3.879 BPW, 233.849 B parameters)
llm_load_print_meta: general.name     = Qwen3 235B A22B
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    1.49 MiB
Tensor blk.0.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.15.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.16.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.16.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.17.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.17.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.18.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.18.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.20.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.21.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.21.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.22.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.22.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.23.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.23.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.24.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.24.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.25.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.25.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.26.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.26.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.27.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.27.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:        CPU buffer size = 22618.55 MiB
llm_load_tensors:        CPU buffer size = 37141.03 MiB
llm_load_tensors:        CPU buffer size = 35082.59 MiB
llm_load_tensors:        CPU buffer size =   630.59 MiB
llm_load_tensors:      CUDA0 buffer size = 15822.01 MiB
llm_load_tensors:      CUDA1 buffer size = 16501.00 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 65536
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  3196.02 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =  3196.02 MiB
llama_new_context_with_model: KV self size  = 6392.00 MiB, K (q8_0): 3196.00 MiB, V (q8_0): 3196.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.16 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  2432.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  2502.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =  1088.05 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 214
INFO [                    init] initializing slots | tid="137280187613184" timestamp=1749456113 n_slots=1
INFO [                    init] new slot | tid="137280187613184" timestamp=1749456113 id_slot=0 n_ctx_slot=65536
INFO [                    main] model loaded | tid="137280187613184" timestamp=1749456113
INFO [                    main] chat template | tid="137280187613184" timestamp=1749456113 chat_example="<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n" built_in=true
INFO [                    main] HTTP server listening | tid="137280187613184" timestamp=1749456113 n_threads_http="111" port="10002" hostname="0.0.0.0"
INFO [            update_slots] all slots are idle | tid="137280187613184" timestamp=1749456113
INFO [      log_server_request] request | tid="137114565677056" timestamp=1749456113 remote_addr="172.17.0.3" remote_port=37714 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="137278110289920" timestamp=1749456119 remote_addr="172.17.0.3" remote_port=59290 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="137280187613184" timestamp=1749456119 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="137280187613184" timestamp=1749456119 id_slot=0 id_task=0 p0=0
INFO [           print_timings] prompt eval time     =    8736.29 ms /   438 tokens (   19.95 ms per token,    50.14 tokens per second) | tid="137280187613184" timestamp=1749456450 id_slot=0 id_task=0 t_prompt_processing=8736.289 n_prompt_tokens_processed=438 t_token=19.945865296803653 n_tokens_second=50.135704072976516
INFO [           print_timings] generation eval time =  322177.45 ms /  4313 runs   (   74.70 ms per token,    13.39 tokens per second) | tid="137280187613184" timestamp=1749456450 id_slot=0 id_task=0 t_token_generation=322177.446 n_decoded=4313 t_token=74.69915279387897 n_tokens_second=13.38703268508746
INFO [           print_timings]           total time =  330913.73 ms | tid="137280187613184" timestamp=1749456450 id_slot=0 id_task=0 t_prompt_processing=8736.289 t_token_generation=322177.446 t_total=330913.735
INFO [            update_slots] slot released | tid="137280187613184" timestamp=1749456450 id_slot=0 id_task=0 n_ctx=65536 n_past=4750 n_system_tokens=0 n_cache_tokens=4750 truncated=false
INFO [            update_slots] all slots are idle | tid="137280187613184" timestamp=1749456450
INFO [      log_server_request] request | tid="137278026412032" timestamp=1749456450 remote_addr="172.17.0.3" remote_port=59306 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="137280187613184" timestamp=1749456450

```

---

👤 **mtcl** commented the **2025-06-09** at **08:11:45**:<br>

And this is when I sent 20K+ prompts to this, like this is insane for me! 329 tokens/second prefill and 13-14 tk/second on token generation. 

```
NFO [            update_slots] kv cache rm [p0, end) | tid="137280187613184" timestamp=1749456577 id_slot=0 id_task=4315 p0=4099
INFO [            update_slots] kv cache rm [p0, end) | tid="137280187613184" timestamp=1749456588 id_slot=0 id_task=4315 p0=8195
INFO [            update_slots] kv cache rm [p0, end) | tid="137280187613184" timestamp=1749456599 id_slot=0 id_task=4315 p0=12291
INFO [            update_slots] kv cache rm [p0, end) | tid="137280187613184" timestamp=1749456610 id_slot=0 id_task=4315 p0=16387
INFO [            update_slots] kv cache rm [p0, end) | tid="137280187613184" timestamp=1749456622 id_slot=0 id_task=4315 p0=20483
INFO [           print_timings] prompt eval time     =   66324.27 ms / 21877 tokens (    3.03 ms per token,   329.85 tokens per second) | tid="137280187613184" timestamp=1749456668 id_slot=0 id_task=4315 t_prompt_processing=66324.269 n_prompt_tokens_processed=21877 t_token=3.0316893998263015 n_tokens_second=329.8490933989789
INFO [           print_timings] generation eval time =   35943.26 ms /   476 runs   (   75.51 ms per token,    13.24 tokens per second) | tid="137280187613184" timestamp=1749456668 id_slot=0 id_task=4315 t_token_generation=35943.258 n_decoded=476 t_token=75.5110462184874 n_tokens_second=13.243095547988442
INFO [           print_timings]           total time =  102267.53 ms | tid="137280187613184" timestamp=1749456668 id_slot=0 id_task=4315 t_prompt_processing=66324.269 t_token_generation=35943.258 t_total=102267.527
INFO [            update_slots] slot released | tid="137280187613184" timestamp=1749456668 id_slot=0 id_task=4315 n_ctx=65536 n_past=22355 n_system_tokens=0 n_cache_tokens=22355 truncated=false
INFO [            update_slots] all slots are idle | tid="137280187613184" timestamp=1749456668
INFO [      log_server_request] request | tid="137278001233920" timestamp=1749456668 remote_addr="172.17.0.3" remote_port=43228 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="137280187613184" timestamp=1749456668

```

---

👤 **mtcl** commented the **2025-06-09** at **08:11:45**:<br>

And this is when I sent 20K+ prompts to this:

```
NFO [            update_slots] kv cache rm [p0, end) | tid="137280187613184" timestamp=1749456577 id_slot=0 id_task=4315 p0=4099
INFO [            update_slots] kv cache rm [p0, end) | tid="137280187613184" timestamp=1749456588 id_slot=0 id_task=4315 p0=8195
INFO [            update_slots] kv cache rm [p0, end) | tid="137280187613184" timestamp=1749456599 id_slot=0 id_task=4315 p0=12291
INFO [            update_slots] kv cache rm [p0, end) | tid="137280187613184" timestamp=1749456610 id_slot=0 id_task=4315 p0=16387
INFO [            update_slots] kv cache rm [p0, end) | tid="137280187613184" timestamp=1749456622 id_slot=0 id_task=4315 p0=20483
INFO [           print_timings] prompt eval time     =   66324.27 ms / 21877 tokens (    3.03 ms per token,   329.85 tokens per second) | tid="137280187613184" timestamp=1749456668 id_slot=0 id_task=4315 t_prompt_processing=66324.269 n_prompt_tokens_processed=21877 t_token=3.0316893998263015 n_tokens_second=329.8490933989789
INFO [           print_timings] generation eval time =   35943.26 ms /   476 runs   (   75.51 ms per token,    13.24 tokens per second) | tid="137280187613184" timestamp=1749456668 id_slot=0 id_task=4315 t_token_generation=35943.258 n_decoded=476 t_token=75.5110462184874 n_tokens_second=13.243095547988442
INFO [           print_timings]           total time =  102267.53 ms | tid="137280187613184" timestamp=1749456668 id_slot=0 id_task=4315 t_prompt_processing=66324.269 t_token_generation=35943.258 t_total=102267.527
INFO [            update_slots] slot released | tid="137280187613184" timestamp=1749456668 id_slot=0 id_task=4315 n_ctx=65536 n_past=22355 n_system_tokens=0 n_cache_tokens=22355 truncated=false
INFO [            update_slots] all slots are idle | tid="137280187613184" timestamp=1749456668
INFO [      log_server_request] request | tid="137278001233920" timestamp=1749456668 remote_addr="172.17.0.3" remote_port=43228 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="137280187613184" timestamp=1749456668

```

---

👤 **ikawrakow** commented the **2025-06-09** at **08:23:01**:<br>

You may want to try the `iq4_ks_r4` model from the same @ubergarm HF repository later (this will give you another video for your channel 😄 ). DeepSeek should run with the same command you used for `iq3_k_r4`. For Qwen3 you may need to reduce number of layers (but start with the 12 layers you used here and only reduce if necessary). Prefill is likely to be better. TG is not easy to predict. The `iq4_ks` matrix multiplication kernel is faster, but there is more data to be fetched from RAM, so one needs to try to see what happens.

---

👤 **mtcl** commented the **2025-06-09** at **08:27:17**:<br>

> You may want to try the `iq4_ks_r4` model from the same @ubergarm HF repository later (this will give you another video for your channel 😄 ). DeepSeek should run with the same command you used for `iq3_k_r4`. For Qwen3 you may need to reduce number of layers (but start with the 12 layers you used here and only reduce if necessary). Prefill is likely to be better. TG is not easy to predict. The `iq4_ks` matrix multiplication kernel is faster, but there is more data to be fetched from RAM, so one needs to try to see what happens.

Download queued and I'll 100% advertise ik_llama! Thank you for being an amazing dev and being so communicative!


If I want to make my own quants, is there a guide out there for it? Like I want to use the Qwen3-235B's IQ4_k_r4 but I can't find it anywhere.

---

👤 **ikawrakow** commented the **2025-06-09** at **08:38:44**:<br>

To make you own quants, you need an imatrix file. You can get those from ubergarm, Bartowsi or Unsloth. Then you use
```
./bin/llama-quantize --imatrix the_imatrix_file --custom-q "..." f16_model_file quantized_model_file Q
```
where `Q` is one of the available quantization types. The secret sauce is what you put into the `--custom-q` argument. If you omit `--custom-q`, you will get some default mix of quants that depends on `Q`. With `--custom-q` you can override specific tensor names to a different quantization type. The syntax is very similar to `-ot`, but instead of having `=CUDA0` or `=CPU`, you put the quantization type. Example:
```
--custom-q "attn=q8_0"
```
will change all tensors with names that match the regular expression `attn` to use `Q8_0` as their quantization type.

`ik_llama.cpp` has many more quantization types than mainline `llama.cpp` (quantization is one of my hobbies), so you need to learn what all these types are before you can become a good "quant cook".

---

👤 **mtcl** commented the **2025-06-09** at **09:10:42**:<br>

Thank you for the details on the Quants!

Quick question, would you know why this crashed? 

```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-server \
    --model /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf \
    --alias ubergarm/DeepSeek-R1-0528-GGUF \
    --ctx-size 40960 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 4096 -ub 4096 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    -ot "blk\.(3)\.ffn_.*=CUDA0" \
    -ot "blk\.(5)\.ffn_.*=CUDA1" \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```

```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
INFO [                    main] build info | tid="131227927433216" timestamp=1749458987 build=3737 commit="58f08e43"
INFO [                    main] system info | tid="131227927433216" timestamp=1749458987 n_threads=57 n_threads_batch=-1 total_threads=112 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 6 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 52 key-value pairs and 1147 tensors from /media/mukul/backup/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ3_K_R4/DeepSeek-R1-0528-IQ3_K_R4-00001-of-00007.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 0528
llama_model_loader: - kv   3:                            general.version str              = 0528
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek-R1
llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
llama_model_loader: - kv   6:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   7:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   8:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   9:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  10:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  11:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  12:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  13: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  15:                          general.file_type u32              = 339
llama_model_loader: - kv  16:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  17:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  18:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  19:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  20:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  21:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  22:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  23:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  24:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  25:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  26:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  27:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  28:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  29:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  30:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  31: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  32: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  33:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  34:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  36:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  37:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  38:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  39:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  40:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  41:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  42:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  43:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  44:               general.quantization_version u32              = 2
llama_model_loader: - kv  45:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/DeepSeek-R1...
llama_model_loader: - kv  46:                   quantize.imatrix.dataset str              = ubergarm-imatrix-calibration-corpus-v...
llama_model_loader: - kv  47:             quantize.imatrix.entries_count i32              = 721
llama_model_loader: - kv  48:              quantize.imatrix.chunks_count i32              = 812
llama_model_loader: - kv  49:                                   split.no u16              = 0
llama_model_loader: - kv  50:                                split.count u16              = 7
llama_model_loader: - kv  51:                        split.tensors.count i32              = 1147
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq3_k_r4:  116 tensors
llama_model_loader: - type iq4_ks_r4:   58 tensors
llm_load_vocab: special tokens cache size = 818
llm_load_vocab: token to piece cache size = 0.8223 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 129280
llm_load_print_meta: n_merges         = 127741
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 163840
llm_load_print_meta: n_embd           = 7168
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 128
llm_load_print_meta: n_head_kv        = 128
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 24576
llm_load_print_meta: n_embd_v_gqa     = 16384
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 18432
llm_load_print_meta: n_expert         = 256
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 0.025
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = IQ3_K_R4 - 3.4325 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 300.938 GiB (3.847 BPW) 
llm_load_print_meta: repeating layers = 299.104 GiB (3.834 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1 0528
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: LF token         = 131 'Ä'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_layer_dense_lead   = 3
llm_load_print_meta: n_lora_q             = 1536
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 2048
llm_load_print_meta: n_expert_shared      = 1
llm_load_print_meta: expert_weights_scale = 2.5
llm_load_print_meta: expert_weights_norm  = 1
llm_load_print_meta: expert_gating_func   = sigmoid
llm_load_print_meta: rope_yarn_log_mul    = 0.1000
llm_load_tensors: ggml ctx size =    1.40 MiB
Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 36486.67 MiB
llm_load_tensors:        CPU buffer size = 43905.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 43905.23 MiB
llm_load_tensors:        CPU buffer size = 43534.23 MiB
llm_load_tensors:        CPU buffer size = 44473.21 MiB
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 13995.99 MiB
llm_load_tensors:      CUDA1 buffer size = 13730.03 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 40960
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =   741.11 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   717.20 MiB
llama_new_context_with_model: KV self size  = 1458.28 MiB, c^KV (q8_0): 1458.28 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  4792.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  4560.02 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   752.05 MiB
llama_new_context_with_model: graph nodes  = 13613
llama_new_context_with_model: graph splits = 149
INFO [                    init] initializing slots | tid="131227927433216" timestamp=1749459186 n_slots=1
INFO [                    init] new slot | tid="131227927433216" timestamp=1749459186 id_slot=0 n_ctx_slot=40960
INFO [                    main] model loaded | tid="131227927433216" timestamp=1749459186
INFO [                    main] chat template | tid="131227927433216" timestamp=1749459186 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
INFO [                    main] HTTP server listening | tid="131227927433216" timestamp=1749459186 n_threads_http="111" port="10002" hostname="0.0.0.0"
INFO [            update_slots] all slots are idle | tid="131227927433216" timestamp=1749459186
INFO [      log_server_request] request | tid="131225838673920" timestamp=1749459190 remote_addr="172.17.0.3" remote_port=55766 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="131225830281216" timestamp=1749459203 remote_addr="172.17.0.3" remote_port=48344 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="131225746403328" timestamp=1749459204 remote_addr="172.17.0.3" remote_port=48354 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="131225738010624" timestamp=1749459210 remote_addr="172.17.0.3" remote_port=60102 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="131227927433216" timestamp=1749459210 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="131227927433216" timestamp=1749459211 id_slot=0 id_task=0 p0=0
INFO [            update_slots] kv cache rm [p0, end) | tid="131227927433216" timestamp=1749459246 id_slot=0 id_task=0 p0=4096
INFO [            update_slots] kv cache rm [p0, end) | tid="131227927433216" timestamp=1749459282 id_slot=0 id_task=0 p0=8192
INFO [            update_slots] kv cache rm [p0, end) | tid="131227927433216" timestamp=1749459319 id_slot=0 id_task=0 p0=12288
INFO [            update_slots] kv cache rm [p0, end) | tid="131227927433216" timestamp=1749459356 id_slot=0 id_task=0 p0=16384
INFO [            update_slots] kv cache rm [p0, end) | tid="131227927433216" timestamp=1749459395 id_slot=0 id_task=0 p0=20480
INFO [            update_slots] kv cache rm [p0, end) | tid="131227927433216" timestamp=1749459434 id_slot=0 id_task=0 p0=24576
INFO [            update_slots] kv cache rm [p0, end) | tid="131227927433216" timestamp=1749459474 id_slot=0 id_task=0 p0=28672
INFO [            update_slots] kv cache rm [p0, end) | tid="131227927433216" timestamp=1749459514 id_slot=0 id_task=0 p0=32768
INFO [            update_slots] slot context shift | tid="131227927433216" timestamp=1749460107 id_slot=0 id_task=0 n_keep=1 n_left=40958 n_discard=20479 n_ctx=40960 n_past=40959 n_system_tokens=0 n_cache_tokens=40959
/home/mukul/dev-ai/ik_llama.cpp/src/llama.cpp:18425: Deepseek2 does not support K-shift
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
Aborted (core dumped)
```

---

👤 **mtcl** commented the **2025-06-09** at **09:11:49**:<br>

once it hit the max context, instead of gracefully stopping, it quit.

---

👤 **ikawrakow** commented the **2025-06-09** at **09:16:26**:<br>

As the message tells you, context shifting is not supported for DeepSeek. You started the server with a context of 40960 tokens (which is the Qwen3 maximum context size), and then tried to have a prompt with more than 40k tokens.

---

👤 **ikawrakow** commented the **2025-06-09** at **09:18:53**:<br>

It is more tricky to do context shifting with MLA, so that's not implemented.

---

👤 **saood06** commented the **2025-06-09** at **09:19:00**:<br>

> once it hit the max context, instead of gracefully stopping, it quit.

Yes. That is a bug with models that don't work with K-shift (I forget why K-shift isn't supported for Deepseek?). It really shouldn't crash, it should gracefully just stop processing. Crash means you can't use it for cached tokens, either to remove context and continue a conversation at an earlier point or saving.

I'm not sure if mainline ever fixed it, but I am just very careful about not hitting it. (My frontend uses the tokenize endpoint so I can know exactly how many tokens I am at all times [even before I send it])

---

👤 **mtcl** commented the **2025-06-09** at **09:23:43**:<br>

OK, i downloaded the IQ4 of @ubergarm  and ran this, it worked with shorter prompt, but with 10K prompt it failed for me, can you please take a look here too?

```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-server \
    --model /home/mukul/dev-ai/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ4_KS_R4/DeepSeek-R1-0528-IQ4_KS_R4-00001-of-00009.gguf \
    --alias ubergarm/DeepSeek-R1-0528-IQ4_KS_R4 \
    --ctx-size 40960 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 4096 -ub 4096 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    -ot "blk\.(3)\.ffn_.*=CUDA0" \
    -ot "blk\.(5)\.ffn_.*=CUDA1" \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```

```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
INFO [                    main] build info | tid="133099125088256" timestamp=1749460517 build=3737 commit="58f08e43"
INFO [                    main] system info | tid="133099125088256" timestamp=1749460517 n_threads=57 n_threads_batch=-1 total_threads=112 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 8 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 52 key-value pairs and 1147 tensors from /home/mukul/dev-ai/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ4_KS_R4/DeepSeek-R1-0528-IQ4_KS_R4-00001-of-00009.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 0528
llama_model_loader: - kv   3:                            general.version str              = 0528
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek-R1
llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
llama_model_loader: - kv   6:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   7:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   8:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   9:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  10:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  11:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  12:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  13: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  15:                          general.file_type u32              = 345
llama_model_loader: - kv  16:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  17:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  18:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  19:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  20:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  21:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  22:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  23:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  24:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  25:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  26:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  27:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  28:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  29:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  30:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  31: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  32: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  33:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  34:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  36:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  37:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  38:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  39:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  40:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  41:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  42:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  43:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  44:               general.quantization_version u32              = 2
llama_model_loader: - kv  45:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/DeepSeek-R1...
llama_model_loader: - kv  46:                   quantize.imatrix.dataset str              = ubergarm-imatrix-calibration-corpus-v...
llama_model_loader: - kv  47:             quantize.imatrix.entries_count i32              = 721
llama_model_loader: - kv  48:              quantize.imatrix.chunks_count i32              = 812
llama_model_loader: - kv  49:                                   split.no u16              = 0
llama_model_loader: - kv  50:                                split.count u16              = 9
llama_model_loader: - kv  51:                        split.tensors.count i32              = 1147
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq4_ks_r4:  116 tensors
llama_model_loader: - type iq5_ks_r4:   58 tensors
llm_load_vocab: special tokens cache size = 818
llm_load_vocab: token to piece cache size = 0.8223 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 129280
llm_load_print_meta: n_merges         = 127741
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 163840
llm_load_print_meta: n_embd           = 7168
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 128
llm_load_print_meta: n_head_kv        = 128
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 24576
llm_load_print_meta: n_embd_v_gqa     = 16384
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 18432
llm_load_print_meta: n_expert         = 256
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 0.025
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = IQ4_KS_R4 - 4.25 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 367.774 GiB (4.701 BPW) 
llm_load_print_meta: repeating layers = 365.940 GiB (4.690 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1 0528
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: LF token         = 131 'Ä'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_layer_dense_lead   = 3
llm_load_print_meta: n_lora_q             = 1536
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 2048
llm_load_print_meta: n_expert_shared      = 1
llm_load_print_meta: expert_weights_scale = 2.5
llm_load_print_meta: expert_weights_norm  = 1
llm_load_print_meta: expert_gating_func   = sigmoid
llm_load_print_meta: rope_yarn_log_mul    = 0.1000
llm_load_tensors: ggml ctx size =    1.40 MiB
Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.5.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 31888.11 MiB
llm_load_tensors:        CPU buffer size = 42582.45 MiB
llm_load_tensors:        CPU buffer size = 40481.67 MiB
llm_load_tensors:        CPU buffer size = 42840.67 MiB
llm_load_tensors:        CPU buffer size = 40481.67 MiB
llm_load_tensors:        CPU buffer size = 42840.67 MiB
llm_load_tensors:        CPU buffer size = 40481.67 MiB
llm_load_tensors:        CPU buffer size = 42840.67 MiB
llm_load_tensors:        CPU buffer size = 41420.65 MiB
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 15175.99 MiB
llm_load_tensors:      CUDA1 buffer size = 14910.03 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 40960
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =   741.11 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   717.20 MiB
llama_new_context_with_model: KV self size  = 1458.28 MiB, c^KV (q8_0): 1458.28 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  6698.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  4560.02 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   752.05 MiB
llama_new_context_with_model: graph nodes  = 13613
llama_new_context_with_model: graph splits = 149
INFO [                    init] initializing slots | tid="133099125088256" timestamp=1749460787 n_slots=1
INFO [                    init] new slot | tid="133099125088256" timestamp=1749460787 id_slot=0 n_ctx_slot=40960
INFO [                    main] model loaded | tid="133099125088256" timestamp=1749460787
INFO [                    main] chat template | tid="133099125088256" timestamp=1749460787 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
INFO [                    main] HTTP server listening | tid="133099125088256" timestamp=1749460787 n_threads_http="111" port="10002" hostname="0.0.0.0"
INFO [            update_slots] all slots are idle | tid="133099125088256" timestamp=1749460787
INFO [      log_server_request] request | tid="133096867356672" timestamp=1749460797 remote_addr="172.17.0.3" remote_port=44752 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="133096858963968" timestamp=1749460811 remote_addr="172.17.0.3" remote_port=49658 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="133099125088256" timestamp=1749460811 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="133099125088256" timestamp=1749460811 id_slot=0 id_task=0 p0=0
INFO [           print_timings] prompt eval time     =     950.72 ms /    10 tokens (   95.07 ms per token,    10.52 tokens per second) | tid="133099125088256" timestamp=1749460848 id_slot=0 id_task=0 t_prompt_processing=950.724 n_prompt_tokens_processed=10 t_token=95.0724 n_tokens_second=10.51829973788397
INFO [           print_timings] generation eval time =   36288.00 ms /   341 runs   (  106.42 ms per token,     9.40 tokens per second) | tid="133099125088256" timestamp=1749460848 id_slot=0 id_task=0 t_token_generation=36287.999 n_decoded=341 t_token=106.41641935483872 n_tokens_second=9.397046114336586
INFO [           print_timings]           total time =   37238.72 ms | tid="133099125088256" timestamp=1749460848 id_slot=0 id_task=0 t_prompt_processing=950.724 t_token_generation=36287.999 t_total=37238.723000000005
INFO [            update_slots] slot released | tid="133099125088256" timestamp=1749460848 id_slot=0 id_task=0 n_ctx=40960 n_past=350 n_system_tokens=0 n_cache_tokens=350 truncated=false
INFO [            update_slots] all slots are idle | tid="133099125088256" timestamp=1749460848
INFO [      log_server_request] request | tid="133096850571264" timestamp=1749460848 remote_addr="172.17.0.3" remote_port=49674 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="133099125088256" timestamp=1749460848
INFO [      log_server_request] request | tid="133096842178560" timestamp=1749460923 remote_addr="172.17.0.3" remote_port=33880 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="133099125088256" timestamp=1749460925 id_slot=0 id_task=343
INFO [            update_slots] kv cache rm [p0, end) | tid="133099125088256" timestamp=1749460925 id_slot=0 id_task=343 p0=2
CUDA error: out of memory
  current device: 0, in function alloc at /home/mukul/dev-ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:384
  cuMemCreate(&handle, reserve_size, &prop, 0)
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
Aborted (core dumped)
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ 
```

---

👤 **ikawrakow** commented the **2025-06-09** at **09:29:48**:<br>

Remove
```
   -ot "blk\.(3)\.ffn_.*=CUDA0" \
   -ot "blk\.(5)\.ffn_.*=CUDA1" \
```

---

👤 **ikawrakow** commented the **2025-06-09** at **09:38:22**:<br>

> I'm not sure if mainline ever fixed it, but I am just very careful about not hitting it. (My frontend uses the tokenize endpoint so I can know exactly how many tokens I am at all times [even before I send it])

I don't know anything about the new unified cache in mainline, and maybe I'm not looking carefully enough, but I don't see K-shift working for DeepSeek. I don't see anything handling the not RoPE'd part of the K-cache, so it does not look like it will work. But it seems the check for DeepSeek arch has been lost in the rewrite.

Strange that there are no issues related to that. Maybe I'm just missing something and it does work? Or maybe it is just that with the mainline snail speed it is very hard to arrive at the situation where context shifting is needed.

---

👤 **mtcl** commented the **2025-06-09** at **09:39:11**:<br>

I was also able to do 32k with -b 2048 -ub 2048. I will try your option after this

```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-server \
    --model /home/mukul/dev-ai/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ4_KS_R4/DeepSeek-R1-0528-IQ4_KS_R4-00001-of-00009.gguf \
    --alias ubergarm/DeepSeek-R1-0528-IQ4_KS_R4 \
    --ctx-size 32768 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 2048 -ub 2048 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    -ot "blk\.(3)\.ffn_.*=CUDA0" \
    -ot "blk\.(5)\.ffn_.*=CUDA1" \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```

```

llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 31888.11 MiB
llm_load_tensors:        CPU buffer size = 42582.45 MiB
llm_load_tensors:        CPU buffer size = 40481.67 MiB
llm_load_tensors:        CPU buffer size = 42840.67 MiB
llm_load_tensors:        CPU buffer size = 40481.67 MiB
llm_load_tensors:        CPU buffer size = 42840.67 MiB
llm_load_tensors:        CPU buffer size = 40481.67 MiB
llm_load_tensors:        CPU buffer size = 42840.67 MiB
llm_load_tensors:        CPU buffer size = 41420.65 MiB
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 15175.99 MiB
llm_load_tensors:      CUDA1 buffer size = 14910.03 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 2048
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =   592.89 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   573.76 MiB
llama_new_context_with_model: KV self size  = 1166.62 MiB, c^KV (q8_0): 1166.62 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  4252.01 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  3560.02 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   312.02 MiB
llama_new_context_with_model: graph nodes  = 8245
llama_new_context_with_model: graph splits = 149
INFO [                    init] initializing slots | tid="140240679325696" timestamp=1749461373 n_slots=1
INFO [                    init] new slot | tid="140240679325696" timestamp=1749461373 id_slot=0 n_ctx_slot=32768
INFO [                    main] model loaded | tid="140240679325696" timestamp=1749461373
INFO [                    main] chat template | tid="140240679325696" timestamp=1749461373 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
INFO [                    main] HTTP server listening | tid="140240679325696" timestamp=1749461373 n_threads_http="111" port="10002" hostname="0.0.0.0"
INFO [            update_slots] all slots are idle | tid="140240679325696" timestamp=1749461373
INFO [      log_server_request] request | tid="140061800480768" timestamp=1749461373 remote_addr="172.17.0.3" remote_port=33158 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="140238584270848" timestamp=1749461383 remote_addr="172.17.0.3" remote_port=43538 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="140238575878144" timestamp=1749461386 remote_addr="172.17.0.3" remote_port=43550 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="140240679325696" timestamp=1749461387 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="140240679325696" timestamp=1749461387 id_slot=0 id_task=0 p0=0
INFO [            update_slots] kv cache rm [p0, end) | tid="140240679325696" timestamp=1749461424 id_slot=0 id_task=0 p0=2048
INFO [            update_slots] kv cache rm [p0, end) | tid="140240679325696" timestamp=1749461462 id_slot=0 id_task=0 p0=4096
INFO [            update_slots] kv cache rm [p0, end) | tid="140240679325696" timestamp=1749461500 id_slot=0 id_task=0 p0=6144
INFO [           print_timings] prompt eval time     =  150323.73 ms /  7074 tokens (   21.25 ms per token,    47.06 tokens per second) | tid="140240679325696" timestamp=1749461640 id_slot=0 id_task=0 t_prompt_processing=150323.728 n_prompt_tokens_processed=7074 t_token=21.25017359344077 n_tokens_second=47.05843910417123
INFO [           print_timings] generation eval time =  103095.45 ms /   934 runs   (  110.38 ms per token,     9.06 tokens per second) | tid="140240679325696" timestamp=1749461640 id_slot=0 id_task=0 t_token_generation=103095.446 n_decoded=934 t_token=110.38056316916487 n_tokens_second=9.059566025835904
INFO [           print_timings]           total time =  253419.17 ms | tid="140240679325696" timestamp=1749461640 id_slot=0 id_task=0 t_prompt_processing=150323.728 t_token_generation=103095.446 t_total=253419.174
INFO [            update_slots] slot released | tid="140240679325696" timestamp=1749461640 id_slot=0 id_task=0 n_ctx=32768 n_past=8007 n_system_tokens=0 n_cache_tokens=8007 truncated=false
INFO [            update_slots] all slots are idle | tid="140240679325696" timestamp=1749461640
INFO [      log_server_request] request | tid="140238567485440" timestamp=1749461640 remote_addr="172.17.0.3" remote_port=43566 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="140240679325696" timestamp=1749461640
```

---

👤 **mtcl** commented the **2025-06-09** at **09:42:24**:<br>

And this is with 64K context. 

```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-server \
    --model /home/mukul/dev-ai/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ4_KS_R4/DeepSeek-R1-0528-IQ4_KS_R4-00001-of-00009.gguf \
    --alias ubergarm/DeepSeek-R1-0528-IQ4_KS_R4 \
    --ctx-size 65536 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 4096 -ub 4096 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```

```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
INFO [                    main] build info | tid="128557749198848" timestamp=1749461746 build=3737 commit="58f08e43"
INFO [                    main] system info | tid="128557749198848" timestamp=1749461746 n_threads=57 n_threads_batch=-1 total_threads=112 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 8 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 52 key-value pairs and 1147 tensors from /home/mukul/dev-ai/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ4_KS_R4/DeepSeek-R1-0528-IQ4_KS_R4-00001-of-00009.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 0528
llama_model_loader: - kv   3:                            general.version str              = 0528
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek-R1
llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
llama_model_loader: - kv   6:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   7:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   8:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   9:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  10:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  11:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  12:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  13: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  15:                          general.file_type u32              = 345
llama_model_loader: - kv  16:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  17:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  18:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  19:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  20:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  21:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  22:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  23:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  24:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  25:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  26:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  27:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  28:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  29:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  30:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  31: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  32: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  33:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  34:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  36:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  37:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  38:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  39:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  40:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  41:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  42:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  43:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  44:               general.quantization_version u32              = 2
llama_model_loader: - kv  45:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/DeepSeek-R1...
llama_model_loader: - kv  46:                   quantize.imatrix.dataset str              = ubergarm-imatrix-calibration-corpus-v...
llama_model_loader: - kv  47:             quantize.imatrix.entries_count i32              = 721
llama_model_loader: - kv  48:              quantize.imatrix.chunks_count i32              = 812
llama_model_loader: - kv  49:                                   split.no u16              = 0
llama_model_loader: - kv  50:                                split.count u16              = 9
llama_model_loader: - kv  51:                        split.tensors.count i32              = 1147
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq4_ks_r4:  116 tensors
llama_model_loader: - type iq5_ks_r4:   58 tensors
llm_load_vocab: special tokens cache size = 818
llm_load_vocab: token to piece cache size = 0.8223 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 129280
llm_load_print_meta: n_merges         = 127741
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 163840
llm_load_print_meta: n_embd           = 7168
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 128
llm_load_print_meta: n_head_kv        = 128
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 24576
llm_load_print_meta: n_embd_v_gqa     = 16384
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 18432
llm_load_print_meta: n_expert         = 256
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 0.025
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = IQ4_KS_R4 - 4.25 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 367.774 GiB (4.701 BPW) 
llm_load_print_meta: repeating layers = 365.940 GiB (4.690 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1 0528
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: LF token         = 131 'Ä'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_layer_dense_lead   = 3
llm_load_print_meta: n_lora_q             = 1536
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 2048
llm_load_print_meta: n_expert_shared      = 1
llm_load_print_meta: expert_weights_scale = 2.5
llm_load_print_meta: expert_weights_norm  = 1
llm_load_print_meta: expert_gating_func   = sigmoid
llm_load_print_meta: rope_yarn_log_mul    = 0.1000
llm_load_tensors: ggml ctx size =    1.40 MiB
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 38317.39 MiB
llm_load_tensors:        CPU buffer size = 42582.45 MiB
llm_load_tensors:        CPU buffer size = 40481.67 MiB
llm_load_tensors:        CPU buffer size = 42840.67 MiB
llm_load_tensors:        CPU buffer size = 40481.67 MiB
llm_load_tensors:        CPU buffer size = 42840.67 MiB
llm_load_tensors:        CPU buffer size = 40481.67 MiB
llm_load_tensors:        CPU buffer size = 42840.67 MiB
llm_load_tensors:        CPU buffer size = 41420.65 MiB
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size =  9056.64 MiB
llm_load_tensors:      CUDA1 buffer size =  8687.38 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 65536
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =  1185.77 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =  1147.51 MiB
llama_new_context_with_model: KV self size  = 2333.25 MiB, c^KV (q8_0): 2333.25 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  7688.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  6992.03 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =  1136.05 MiB
llama_new_context_with_model: graph nodes  = 13613
llama_new_context_with_model: graph splits = 149
INFO [                    init] initializing slots | tid="128557749198848" timestamp=1749461834 n_slots=1
INFO [                    init] new slot | tid="128557749198848" timestamp=1749461834 id_slot=0 n_ctx_slot=65536
INFO [                    main] model loaded | tid="128557749198848" timestamp=1749461834
INFO [                    main] chat template | tid="128557749198848" timestamp=1749461834 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
INFO [                    main] HTTP server listening | tid="128557749198848" timestamp=1749461834 n_threads_http="111" port="10002" hostname="0.0.0.0"
INFO [            update_slots] all slots are idle | tid="128557749198848" timestamp=1749461834
INFO [      log_server_request] request | tid="128291451105280" timestamp=1749461834 remote_addr="172.17.0.3" remote_port=44606 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="128291442712576" timestamp=1749461834 remote_addr="172.17.0.3" remote_port=44614 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="128555551813632" timestamp=1749461843 remote_addr="172.17.0.3" remote_port=51610 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="128555543420928" timestamp=1749461855 remote_addr="172.17.0.3" remote_port=39760 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="128557749198848" timestamp=1749461855 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="128557749198848" timestamp=1749461855 id_slot=0 id_task=0 p0=0
INFO [            update_slots] kv cache rm [p0, end) | tid="128557749198848" timestamp=1749461910 id_slot=0 id_task=0 p0=4096


INFO [           print_timings] prompt eval time     =  109691.87 ms /  7074 tokens (   15.51 ms per token,    64.49 tokens per second) | tid="128557749198848" timestamp=1749462092 id_slot=0 id_task=0 t_prompt_processing=109691.87 n_prompt_tokens_processed=7074 t_token=15.506342945999434 n_tokens_second=64.48973839173314
INFO [           print_timings] generation eval time =  127046.30 ms /  1118 runs   (  113.64 ms per token,     8.80 tokens per second) | tid="128557749198848" timestamp=1749462092 id_slot=0 id_task=0 t_token_generation=127046.301 n_decoded=1118 t_token=113.63712075134168 n_tokens_second=8.799941369406733
INFO [           print_timings]           total time =  236738.17 ms | tid="128557749198848" timestamp=1749462092 id_slot=0 id_task=0 t_prompt_processing=109691.87 t_token_generation=127046.301 t_total=236738.171
INFO [            update_slots] slot released | tid="128557749198848" timestamp=1749462092 id_slot=0 id_task=0 n_ctx=65536 n_past=8191 n_system_tokens=0 n_cache_tokens=8191 truncated=false
INFO [            update_slots] all slots are idle | tid="128557749198848" timestamp=1749462092
INFO [      log_server_request] request | tid="128555535028224" timestamp=1749462092 remote_addr="172.17.0.3" remote_port=39770 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="128557749198848" timestamp=1749462092
```

---

👤 **saood06** commented the **2025-06-09** at **09:42:30**:<br>

>Strange that there are no issues related to that. Maybe I'm just missing something and it does work? Or maybe it is just that with the mainline snail speed it is very hard to arrive at the situation where context shifting is needed.

I thought there was in mainline, this has been an issue for as long as I can remember, but once I experienced it, I just became vigilant about not hitting the limit. (why I liked #290 as it is a QoL feature as you can overallocate KV cache without being punished with a crash, just degraded performance once you cross the threshold)

---

👤 **saood06** commented the **2025-06-09** at **09:42:30**:<br>

>Strange that there are no issues related to that. Maybe I'm just missing something and it does work? Or maybe it is just that with the mainline snail speed it is very hard to arrive at the situation where context shifting is needed.

I thought there was in mainline, this has been an issue for as long as I can remember, but once I experienced it, I just became vigilant about not hitting the limit. (#290 is a QoL feature as you can overallocate KV cache without being punished with a crash, just degraded performance)

---

👤 **ubergarm** commented the **2025-06-09** at **14:22:17**:<br>

@mtcl 

Looks like y'all had a busy day! Glad to see you managed to achieve much improved speeds learning the commands to match your hardware.

If you want a more visible and understandable benchmark of speeds for a given configuration, you can change out the binary from `llama-server` to `llama-sweep-bench` and run it e.g.:

```bash
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-sweep-bench \
    --model /home/mukul/dev-ai/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ4_KS_R4/DeepSeek-R1-0528-IQ4_KS_R4-00001-of-00009.gguf \
    --alias ubergarm/DeepSeek-R1-0528-IQ4_KS_R4 \
    --ctx-size 65536 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 4096 -ub 4096 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```

You can remove `--alias` `--parallel` `--host` and `--port` but its probably fine to just leave them there as well (to keep it simple) as they are not used for `llama-sweep-bench`.

Then you can see how the speed drops with longer context and better understand the consequences of long context etc.

> Would you be able to post a guide on how to make the IQ4 version of the Qwen Model?

I have posted a [quant cookers guide here](https://github.com/ikawrakow/ik_llama.cpp/discussions/434) to help people get started and show some examples. As ik mentions, feel free to use an existing imatrix file from myself, unsloth, or bartowski etc. Or you can make your own using the instructions I provided.

If you check my huggingface repo's I list some of my "secret recipes", which you can use as a starting point for your mixes.

The guide does not discuss how to convert DeepSeek fp8 to bf16 GGUF. That is an extra first step only for DeepSeek safetensors. You can find some of that buried in my original guide in a details fold about `triton-cpu` and the evshiron llama.cpp fork. Give you have newer GPUs you might be able to cast it from fp8 directly on GPU with the "normal" way, but I've never done that myself.

Enjoy your setup and new GPUs and good luck with your latest videos!

---

👤 **ubergarm** commented the **2025-06-09** at **14:22:17**:<br>

@mtcl 

Looks like y'all had a busy day! Glad to see you managed to achieve much improved speeds learning the commands to match your hardware.

If you want a more visible and understandable benchmark of speeds for a given configuration, you can change out the binary from `llama-server` to `llama-sweep-bench` and run it e.g.:

```bash
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-sweep-bench \
    --model /home/mukul/dev-ai/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ4_KS_R4/DeepSeek-R1-0528-IQ4_KS_R4-00001-of-00009.gguf \
    --alias ubergarm/DeepSeek-R1-0528-IQ4_KS_R4 \
    --ctx-size 65536 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 4096 -ub 4096 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```

You can remove `--alias` `--parallel` `--host` and `--port` but its probably fine to just leave them there as well (to keep it simple) as they are not used for `llama-sweep-bench`.

Then you can see how the speed drops with longer context and better understand the consequences of long context etc.

> Would you be able to post a guide on how to make the IQ4 version of the Qwen Model?

I have posted a [quant cookers guide here](https://github.com/ikawrakow/ik_llama.cpp/discussions/434) to help people get started and show some examples. As ik mentions, feel free to use an existing imatrix file from myself, unsloth, or bartowski etc. Or you can make your own using the instructions I provided.

The guide does not discuss how to convert DeepSeek fp8 to bf16 GGUF. That is an extra first step only for DeepSeek safetensors. You can find some of that buried in my original guide in a details fold about `triton-cpu` and the evshiron llama.cpp fork. Give you have newer GPUs you might be able to cast it from fp8 directly on GPU with the "normal" way, but I've never done that myself.

Enjoy your setup and new GPUs and good luck with your latest videos!

---

👤 **ikawrakow** commented the **2025-06-09** at **15:19:10**:<br>

@ubergarm 

Btw, what type did you use for the `output.weight` tensor in these models? The HF model browser does not work with them and  don't feel like downloading 50 GB to check. Or more general, did you use `IQ6_K` for any of your published models?

---

👤 **ubergarm** commented the **2025-06-09** at **15:32:13**:<br>

> Btw, what type did you use for the output.weight tensor in these models? The HF model browser does not work with them and don't feel like downloading 50 GB to check. Or more general, did you use IQ6_K for any of your published models?

Yeah I'm not sure why HF model browser stopped working with some of my quants and complains `Error: not a valid gguf file: not starting with GGUF magic number` for some reason.

None of my recipes are currently using `IQ6_K` for `output.weight` in checking my scripts. For my recent R1-0528's I've used either `q8_0`, `iq5_ks`, or `iq4_ks`.

*EDIT*: Haha, well I just double checked on that incomplete uploaded [ubergarm/DeepSeek-R1T-Chimera-GGUF](https://huggingface.co/ubergarm/DeepSeek-R1T-Chimera-GGUF/tree/main/DeepSeek-R1T-Chimera-IQ4_KS) and apparently I did use `output\.weight=iq6_k`. I'm happy to cancel that and delete it honestly and re-do it and upload from a rig with reasonable uplink if you need to break anything though.

If you need more specifics I can run a gguf-dump on anything.

---

👤 **ikawrakow** commented the **2025-06-09** at **15:39:13**:<br>

No need, thanks. It is just that as we were running these experiments I thought that the compute buffers were larger than I was expecting them to be, and one hypothesis I had was that the output tensor was `IQ6_K` and that `IQ6_K` does not have MMQ, so needs to be dequantized to `f16`, and that increases the compute buffer quite a bit. But I just checked, `IQ6_K` does have MMQ, so that's not it.

---

👤 **mtcl** commented the **2025-06-09** at **19:18:33**:<br>

> [@mtcl](https://github.com/mtcl)

> If you want a more visible and understandable benchmark of speeds for a given configuration, you can change out the binary from `llama-server` to `llama-sweep-bench` and run it e.g.:

> Enjoy your setup and new GPUs and good luck with your latest videos!

Thank you @ubergarm !

Now I need to learn how to read these outputs :)

```
CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-sweep-bench \
    --model /home/mukul/dev-ai/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ4_KS_R4/DeepSeek-R1-0528-IQ4_KS_R4-00001-of-00009.gguf \
    --alias ubergarm/DeepSeek-R1-0528-IQ4_KS_R4 \
    --ctx-size 65536 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 4096 -ub 4096 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```
```
main: n_kv_max = 65536, n_batch = 4096, n_ubatch = 4096, flash_attn = 1, n_gpu_layers = 63, n_threads = 57, n_threads_batch = 57
```

```
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   40.839 |   100.30 |  124.582 |     8.22 |
|  4096 |   1024 |   4096 |   40.847 |   100.28 |  112.796 |     9.08 |
|  4096 |   1024 |   8192 |   41.224 |    99.36 |  116.865 |     8.76 |
|  4096 |   1024 |  12288 |   41.860 |    97.85 |  115.780 |     8.84 |
|  4096 |   1024 |  16384 |   42.717 |    95.89 |  110.798 |     9.24 |
|  4096 |   1024 |  20480 |   43.358 |    94.47 |  119.139 |     8.59 |
|  4096 |   1024 |  24576 |   44.067 |    92.95 |  118.138 |     8.67 |
|  4096 |   1024 |  28672 |   44.897 |    91.23 |  120.028 |     8.53 |
|  4096 |   1024 |  32768 |   46.109 |    88.83 |  116.720 |     8.77 |
|  4096 |   1024 |  36864 |   47.268 |    86.65 |  119.693 |     8.56 |
|  4096 |   1024 |  40960 |   48.326 |    84.76 |  124.217 |     8.24 |
|  4096 |   1024 |  45056 |   47.720 |    85.83 |  122.807 |     8.34 |
|  4096 |   1024 |  49152 |   48.337 |    84.74 |  129.565 |     7.90 |
|  4096 |   1024 |  53248 |   49.039 |    83.53 |  128.600 |     7.96 |
|  4096 |   1024 |  57344 |   49.896 |    82.09 |  119.462 |     8.57 |
|  4096 |   1024 |  61440 |   51.657 |    79.29 |  130.716 |     7.83 |

```

Second test with one GPU
```
CUDA_VISIBLE_DEVICES="1" ./build/bin/llama-sweep-bench     --model /home/mukul/dev-ai/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ4_KS_R4/DeepSeek-R1-0528-IQ4_KS_R4-00001-of-00009.gguf     --alias ubergarm/DeepSeek-R1-0528-IQ4_KS_R4     --ctx-size 16384     -ctk q8_0     -mla 3 -fa     -b 2048 -ub 2048     -amb 512     -fmoe     --n-gpu-layers 63     --override-tensor exps=CPU     --parallel 1     --threads 57     --host 0.0.0.0     --port 10002

main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 2048, flash_attn = 1, n_gpu_layers = 63, n_threads = 57, n_threads_batch = 57

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |   38.661 |    52.97 |   60.806 |     8.42 |
|  2048 |    512 |   2048 |   38.916 |    52.63 |   61.412 |     8.34 |
|  2048 |    512 |   4096 |   40.109 |    51.06 |   61.294 |     8.35 |
|  2048 |    512 |   6144 |   39.816 |    51.44 |   62.676 |     8.17 |
|  2048 |    512 |   8192 |   40.202 |    50.94 |   56.425 |     9.07 |
|  2048 |    512 |  10240 |   41.658 |    49.16 |   56.552 |     9.05 |
|  2048 |    512 |  12288 |   41.141 |    49.78 |   56.748 |     9.02 |
|  2048 |    512 |  14336 |   66.404 |    30.84 |   60.540 |     8.46 |
```


Third test with 2 GPUs
```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-sweep-bench \
    --model /home/mukul/dev-ai/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ4_KS_R4/DeepSeek-R1-0528-IQ4_KS_R4-00001-of-00009.gguf \
    --alias ubergarm/DeepSeek-R1-0528-IQ4_KS_R4 \
    --ctx-size 32768 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 2048 -ub 2048 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    -ot "blk\.(3)\.ffn_.*=CUDA0" \
    -ot "blk\.(5)\.ffn_.*=CUDA1" \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002



main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 2048, flash_attn = 1, n_gpu_layers = 63, n_threads = 57, n_threads_batch = 57

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |   40.820 |    50.17 |   55.687 |     9.19 |
|  2048 |    512 |   2048 |   38.313 |    53.45 |   50.972 |    10.04 |
|  2048 |    512 |   4096 |   38.419 |    53.31 |   52.591 |     9.74 |
|  2048 |    512 |   6144 |   38.707 |    52.91 |   52.099 |     9.83 |
|  2048 |    512 |   8192 |   38.808 |    52.77 |   55.032 |     9.30 |
|  2048 |    512 |  10240 |   37.745 |    54.26 |   58.197 |     8.80 |
|  2048 |    512 |  12288 |   38.116 |    53.73 |   55.813 |     9.17 |
|  2048 |    512 |  14336 |   38.348 |    53.41 |   56.316 |     9.09 |
|  2048 |    512 |  16384 |   39.717 |    51.57 |   57.785 |     8.86 |
|  2048 |    512 |  18432 |   38.704 |    52.91 |   58.078 |     8.82 |
|  2048 |    512 |  20480 |   40.091 |    51.08 |   59.625 |     8.59 |
|  2048 |    512 |  22528 |   39.240 |    52.19 |   55.395 |     9.24 |
|  2048 |    512 |  24576 |   69.900 |    29.30 |   57.348 |     8.93 |
|  2048 |    512 |  26624 |   53.911 |    37.99 |   63.346 |     8.08 |
|  2048 |    512 |  28672 |   40.947 |    50.02 |   54.078 |     9.47 |
|  2048 |    512 |  30720 |   40.169 |    50.98 |   56.996 |     8.98 |
```


Qwen3-235B tests 2 GPUs with -ot parameter

```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0,1" ./build/bin/llama-sweep-bench \
  --model /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/\
Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf \
  --alias ubergarm/Qwen3-235B-A22B-mix-IQ3_K \
  --ctx-size 40960 \
  -ctk q8_0 -ctv q8_0 \
  -fa \
  -b 4096 -ub 4096 \
  -fmoe \
  --n-gpu-layers 100 \
  -ot "blk\.[0-9]\.ffn=CUDA0,blk\.1[0-1]\.ffn=CUDA0" \
  -ot "blk\.1[5-8]\.ffn=CUDA1,blk\.2[0-7]\.ffn=CUDA1" \
  --override-tensor exps=CPU \
  --parallel 1 \
  --threads 56 \
  --host 0.0.0.0 \
  --port 10002


main: n_kv_max = 40960, n_batch = 4096, n_ubatch = 4096, flash_attn = 1, n_gpu_layers = 100, n_threads = 56, n_threads_batch = 56

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   10.698 |   382.88 |   61.007 |    16.78 |
|  4096 |   1024 |   4096 |   10.766 |   380.45 |   61.181 |    16.74 |
|  4096 |   1024 |   8192 |   10.949 |   374.08 |   64.612 |    15.85 |
|  4096 |   1024 |  12288 |   11.183 |   366.29 |   55.323 |    18.51 |
|  4096 |   1024 |  16384 |   11.497 |   356.25 |   70.926 |    14.44 |
|  4096 |   1024 |  20480 |   11.666 |   351.11 |   71.375 |    14.35 |
|  4096 |   1024 |  24576 |   11.899 |   344.24 |   73.816 |    13.87 |
|  4096 |   1024 |  28672 |   12.204 |   335.63 |   75.642 |    13.54 |
|  4096 |   1024 |  32768 |   12.338 |   331.97 |   76.375 |    13.41 |
|  4096 |   1024 |  36864 |   12.595 |   325.22 |   79.639 |    12.86 |
```


Qwen3-235B tests 2 GPUs without -ot parameter
```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-sweep-bench \
    --model /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf \
  --alias ubergarm/Qwen3-235B-A22B-mix-IQ3_K \
    --ctx-size 40960 \
    -ctk q8_0 -ctv q8_0 \
    -fa \
    -b 4096 -ub 4096 \
    -fmoe \
    --n-gpu-layers 100 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 56 \
    --host 0.0.0.0 \
    --port 10002


main: n_kv_max = 40960, n_batch = 4096, n_ubatch = 4096, flash_attn = 1, n_gpu_layers = 100, n_threads = 56, n_threads_batch = 56

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   13.478 |   303.90 |   55.508 |    18.45 |
|  4096 |   1024 |   4096 |   13.614 |   300.87 |   57.259 |    17.88 |
|  4096 |   1024 |   8192 |   13.801 |   296.80 |   59.898 |    17.10 |
|  4096 |   1024 |  12288 |   13.996 |   292.66 |   61.289 |    16.71 |
|  4096 |   1024 |  16384 |   14.244 |   287.56 |   63.258 |    16.19 |
|  4096 |   1024 |  20480 |   14.482 |   282.82 |   64.498 |    15.88 |
|  4096 |   1024 |  24576 |   14.715 |   278.36 |   66.287 |    15.45 |
|  4096 |   1024 |  28672 |   14.911 |   274.69 |   67.805 |    15.10 |
|  4096 |   1024 |  32768 |   15.128 |   270.75 |   70.190 |    14.59 |
|  4096 |   1024 |  36864 |   15.334 |   267.12 |   72.498 |    14.12 |
```

Qwen3-235B tests 1 GPUs without -ot parameter

```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="1" ./build/bin/llama-sweep-bench \
    --model /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf \
  --alias ubergarm/Qwen3-235B-A22B-mix-IQ3_K \
    --ctx-size 40960 \
    -ctk q8_0 -ctv q8_0 \
    -fa \
    -b 4096 -ub 4096 \
    -fmoe \
    --n-gpu-layers 100 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 56 \
    --host 0.0.0.0 \
    --port 10002


main: n_kv_max = 40960, n_batch = 4096, n_ubatch = 4096, flash_attn = 1, n_gpu_layers = 100, n_threads = 56, n_threads_batch = 56

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   13.133 |   311.88 |   55.024 |    18.61 |
|  4096 |   1024 |   4096 |   13.454 |   304.44 |   56.686 |    18.06 |
|  4096 |   1024 |   8192 |   13.733 |   298.25 |   59.303 |    17.27 |
|  4096 |   1024 |  12288 |   14.060 |   291.33 |   60.571 |    16.91 |
|  4096 |   1024 |  16384 |   14.432 |   283.82 |   62.472 |    16.39 |
|  4096 |   1024 |  20480 |   14.898 |   274.93 |   63.748 |    16.06 |
|  4096 |   1024 |  24576 |   15.354 |   266.77 |   65.519 |    15.63 |
|  4096 |   1024 |  28672 |   15.726 |   260.45 |   66.587 |    15.38 |
|  4096 |   1024 |  32768 |   16.086 |   254.64 |   68.739 |    14.90 |
|  4096 |   1024 |  36864 |   16.453 |   248.95 |   71.106 |    14.40 |
```

---

👤 **mtcl** commented the **2025-06-09** at **19:18:33**:<br>

> [@mtcl](https://github.com/mtcl)
> 
> Looks like y'all had a busy day! Glad to see you managed to achieve much improved speeds learning the commands to match your hardware.
> 
> If you want a more visible and understandable benchmark of speeds for a given configuration, you can change out the binary from `llama-server` to `llama-sweep-bench` and run it e.g.:
> 
> (base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-sweep-bench \
>     --model /home/mukul/dev-ai/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ4_KS_R4/DeepSeek-R1-0528-IQ4_KS_R4-00001-of-00009.gguf \
>     --alias ubergarm/DeepSeek-R1-0528-IQ4_KS_R4 \
>     --ctx-size 65536 \
>     -ctk q8_0 \
>     -mla 3 -fa \
>     -b 4096 -ub 4096 \
>     -amb 512 \
>     -fmoe \
>     --n-gpu-layers 63 \
>     --override-tensor exps=CPU \
>     --parallel 1 \
>     --threads 57 \
>     --host 0.0.0.0 \
>     --port 10002
> 
> You can remove `--alias` `--parallel` `--host` and `--port` but its probably fine to just leave them there as well (to keep it simple) as they are not used for `llama-sweep-bench`.
> 
> Then you can see how the speed drops with longer context and better understand the consequences of long context etc.
> 
> > Would you be able to post a guide on how to make the IQ4 version of the Qwen Model?
> 
> I have posted a [quant cookers guide here](https://github.com/ikawrakow/ik_llama.cpp/discussions/434) to help people get started and show some examples. As ik mentions, feel free to use an existing imatrix file from myself, unsloth, or bartowski etc. Or you can make your own using the instructions I provided.
> 
> If you check my huggingface repo's I list some of my "secret recipes", which you can use as a starting point for your mixes.
> 
> The guide does not discuss how to convert DeepSeek fp8 to bf16 GGUF. That is an extra first step only for DeepSeek safetensors. You can find some of that buried in my original guide in a details fold about `triton-cpu` and the evshiron llama.cpp fork. Give you have newer GPUs you might be able to cast it from fp8 directly on GPU with the "normal" way, but I've never done that myself.
> 
> Enjoy your setup and new GPUs and good luck with your latest videos!

Thank you @ubergarm !

Now I need to learn how to read these outputs :)

```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="0, 1" ./build/bin/llama-sweep-bench \
    --model /home/mukul/dev-ai/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ4_KS_R4/DeepSeek-R1-0528-IQ4_KS_R4-00001-of-00009.gguf \
    --alias ubergarm/DeepSeek-R1-0528-IQ4_KS_R4 \
    --ctx-size 65536 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 4096 -ub 4096 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
```
```
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 38317.39 MiB
llm_load_tensors:        CPU buffer size = 42582.45 MiB
llm_load_tensors:        CPU buffer size = 40481.67 MiB
llm_load_tensors:        CPU buffer size = 42840.67 MiB
llm_load_tensors:        CPU buffer size = 40481.67 MiB
llm_load_tensors:        CPU buffer size = 42840.67 MiB
llm_load_tensors:        CPU buffer size = 40481.67 MiB
llm_load_tensors:        CPU buffer size = 42840.67 MiB
llm_load_tensors:        CPU buffer size = 41420.65 MiB
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size =  9056.64 MiB
llm_load_tensors:      CUDA1 buffer size =  8687.38 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 65536
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =  1185.77 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =  1147.51 MiB
llama_new_context_with_model: KV self size  = 2333.25 MiB, c^KV (q8_0): 2333.25 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.49 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  7688.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  6992.03 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =  1136.05 MiB
llama_new_context_with_model: graph nodes  = 13613
llama_new_context_with_model: graph splits = 149

main: n_kv_max = 65536, n_batch = 4096, n_ubatch = 4096, flash_attn = 1, n_gpu_layers = 63, n_threads = 57, n_threads_batch = 57

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   40.839 |   100.30 |  124.582 |     8.22 |
|  4096 |   1024 |   4096 |   40.847 |   100.28 |  112.796 |     9.08 |
|  4096 |   1024 |   8192 |   41.224 |    99.36 |  116.865 |     8.76 |
|  4096 |   1024 |  12288 |   41.860 |    97.85 |  115.780 |     8.84 |
|  4096 |   1024 |  16384 |   42.717 |    95.89 |  110.798 |     9.24 |
|  4096 |   1024 |  20480 |   43.358 |    94.47 |  119.139 |     8.59 |
|  4096 |   1024 |  24576 |   44.067 |    92.95 |  118.138 |     8.67 |
|  4096 |   1024 |  28672 |   44.897 |    91.23 |  120.028 |     8.53 |
|  4096 |   1024 |  32768 |   46.109 |    88.83 |  116.720 |     8.77 |
|  4096 |   1024 |  36864 |   47.268 |    86.65 |  119.693 |     8.56 |
|  4096 |   1024 |  40960 |   48.326 |    84.76 |  124.217 |     8.24 |
|  4096 |   1024 |  45056 |   47.720 |    85.83 |  122.807 |     8.34 |
|  4096 |   1024 |  49152 |   48.337 |    84.74 |  129.565 |     7.90 |
|  4096 |   1024 |  53248 |   49.039 |    83.53 |  128.600 |     7.96 |
|  4096 |   1024 |  57344 |   49.896 |    82.09 |  119.462 |     8.57 |
|  4096 |   1024 |  61440 |   51.657 |    79.29 |  130.716 |     7.83 |
```

---

👤 **mtcl** commented the **2025-06-09** at **20:58:25**:<br>

@ubergarm or @ikawrakow a question for you, if I don't want to cook my own quant, what is the easiest way to find the quant on huggingface that will be most compatible with ik_llama? 

Do all these parameters also work with q4_k_m if i already have some q4_k_m models downloaded and I want to use them with ik_llama?

---

👤 **ubergarm** commented the **2025-06-09** at **21:40:36**:<br>

@mtcl 

You are both enthusiastic and patient! I forgot to suggest adding `-wb` for `--warmup-batch` to get the first datapoint collected more in line, but now you have a useful tool to decide how to run your LLMs depending on your task.

> Now I need to learn how to read these outputs :)

There is a provided python tool to plot the data, or you can vibe code something up easily enough like I did to show your results and compare the configurations.

The left side of the x-axis is smaller kv-cache so simulates the speed working with shorter prompts. The right side of the x-axis is more representative of speeds when working with a longer prompt or ongoing multi-turn chat.

The more tok/sec the merrier!

#### R1-0528

![Image](https://github.com/user-attachments/assets/c95ef8cf-9f6d-4d85-a6d9-28bd551ffd9f)

This suggests using `-ub 4096 -b 4096` gives more prompt processing speed. Your test cases change a few things so I didn't capture everything. I often do a simple A/B test changing only one variable to make it easier to understand how to read the results.

But this quick demo gives you some perspective and shows how you can see the effects of choosing to offload more layers vs increasing batch sizes etc.

#### Qwen3-235B-A22B

![Image](https://github.com/user-attachments/assets/ee4dabc2-5c50-476d-8927-699f838148ff)

Again for Qwen3-235B-A22B you see how the big increase in prompt processing doesn't hurt token generation very much so likely a good trade-off for many applications.

---

👤 **ubergarm** commented the **2025-06-09** at **21:48:00**:<br>

> if I don't want to cook my own quant, what is the easiest way to find the quant on huggingface that will be most compatible with ik_llama?

ik_llama.cpp supports pretty much everything that runs on mainline llama.cpp plus the additional SOTA quants like `iq4_ks` etc. If you want to see quants specific to ik_llama.cpp, myself and some others tend to use the tag [ik_llama.cpp](https://huggingface.co/models?other=ik_llama.cpp) to help people find them easier.

If there is a specific model in which you're interested having a SOTA ik_llama.cpp quant, and you don't want to cook it and release it yourself, you could ask myself or possibly some of the folks on huggingface who have released stuff already.

> Do all these parameters also work with q4_k_m if i already have some q4_k_m models downloaded and I want to use them with ik_llama?

Yes, In general you can run them pretty much the same. There may be some corner cases that arise but it should throw and error and you can ask here. So, yes, go ahead and use your existing quants with ik_llama.cpp and likely you can still get some speed boost with `-rtr` and such. ik is the author of a number of the quants still used regularly on mainline/ollama/lmstudio/etc including the still popular [iq4_xs](https://github.com/ggml-org/llama.cpp/pull/5747).

Cheers!

---

👤 **mtcl** commented the **2025-06-09** at **23:55:27**:<br>

Hmm, I'm struggling with this 5090 and ik_llama not sure what's going wrong here. It works fine with 4090 but crashes on 5090.

---

👤 **mtcl** commented the **2025-06-10** at **00:30:44**:<br>

5090 errors out:
```
CUDA_VISIBLE_DEVICES="1" ./build/bin/llama-server \
    --model /home/mukul/dev-ai/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ4_KS_R4/DeepSeek-R1-0528-IQ4_KS_R4-00001-of-00009.gguf \
    --alias ubergarm/DeepSeek-R1-0528-IQ4_KS_R4 \
    --ctx-size 32768 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 2048 -ub 2048 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
INFO [                    main] build info | tid="123801787162624" timestamp=1749514690 build=3738 commit="fa90a986"
INFO [                    main] system info | tid="123801787162624" timestamp=1749514690 n_threads=57 n_threads_batch=-1 total_threads=112 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 8 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 52 key-value pairs and 1147 tensors from /home/mukul/dev-ai/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ4_KS_R4/DeepSeek-R1-0528-IQ4_KS_R4-00001-of-00009.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 0528
llama_model_loader: - kv   3:                            general.version str              = 0528
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek-R1
llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
llama_model_loader: - kv   6:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   7:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   8:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   9:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  10:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  11:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  12:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  13: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  15:                          general.file_type u32              = 345
llama_model_loader: - kv  16:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  17:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  18:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  19:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  20:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  21:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  22:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  23:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  24:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  25:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  26:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  27:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  28:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  29:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  30:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  31: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  32: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  33:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  34:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�..
llama_model_loader: - kv  36:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  37:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  38:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  39:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  40:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  41:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  42:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  43:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  44:               general.quantization_version u32              = 2
llama_model_loader: - kv  45:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/DeepSeek-R1...
llama_model_loader: - kv  46:                   quantize.imatrix.dataset str              = ubergarm-imatrix-calibration-corpus-v...
llama_model_loader: - kv  47:             quantize.imatrix.entries_count i32              = 721
llama_model_loader: - kv  48:              quantize.imatrix.chunks_count i32              = 812
llama_model_loader: - kv  49:                                   split.no u16              = 0
llama_model_loader: - kv  50:                                split.count u16              = 9
llama_model_loader: - kv  51:                        split.tensors.count i32              = 1147
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq4_ks_r4:  116 tensors
llama_model_loader: - type iq5_ks_r4:   58 tensors
llm_load_vocab: special tokens cache size = 818
llm_load_vocab: token to piece cache size = 0.8223 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 129280
llm_load_print_meta: n_merges         = 127741
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 163840
llm_load_print_meta: n_embd           = 7168
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 128
llm_load_print_meta: n_head_kv        = 128
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 24576
llm_load_print_meta: n_embd_v_gqa     = 16384
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 18432
llm_load_print_meta: n_expert         = 256
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 0.025
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = IQ4_KS_R4 - 4.25 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 367.774 GiB (4.701 BPW) 
llm_load_print_meta: repeating layers = 365.940 GiB (4.690 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1 0528
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: LF token         = 131 'Ä'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_layer_dense_lead   = 3
llm_load_print_meta: n_lora_q             = 1536
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 2048
llm_load_print_meta: n_expert_shared      = 1
llm_load_print_meta: expert_weights_scale = 2.5
llm_load_print_meta: expert_weights_norm  = 1
llm_load_print_meta: expert_gating_func   = sigmoid
llm_load_print_meta: rope_yarn_log_mul    = 0.1000
llm_load_tensors: ggml ctx size =    0.93 MiB
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 38317.39 MiB
llm_load_tensors:        CPU buffer size = 42582.45 MiB
llm_load_tensors:        CPU buffer size = 40481.67 MiB
llm_load_tensors:        CPU buffer size = 42840.67 MiB
llm_load_tensors:        CPU buffer size = 40481.67 MiB
llm_load_tensors:        CPU buffer size = 42840.67 MiB
llm_load_tensors:        CPU buffer size = 40481.67 MiB
llm_load_tensors:        CPU buffer size = 42840.67 MiB
llm_load_tensors:        CPU buffer size = 41420.65 MiB
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 17744.02 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 2048
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =  1166.65 MiB
llama_new_context_with_model: KV self size  = 1166.62 MiB, c^KV (q8_0): 1166.62 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =  4252.01 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   312.02 MiB
llama_new_context_with_model: graph nodes  = 8245
llama_new_context_with_model: graph splits = 118
ggml_cuda_compute_forward: FUSED_RMS_NORM failed
CUDA error: no kernel image is available for execution on the device
  current device: 0, in function ggml_cuda_compute_forward at /home/mukul/dev-ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:2963
  err
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
Aborted (core dumped)
```

but 4090 works:
```
CUDA_VISIBLE_DEVICES="0" ./build/bin/llama-server \
    --model /home/mukul/dev-ai/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ4_KS_R4/DeepSeek-R1-0528-IQ4_KS_R4-00001-of-00009.gguf \
    --alias ubergarm/DeepSeek-R1-0528-IQ4_KS_R4 \
    --ctx-size 32768 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 2048 -ub 2048 \
    -amb 512 \
    -fmoe \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 57 \
    --host 0.0.0.0 \
    --port 10002
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
INFO [                    main] build info | tid="128766126837760" timestamp=1749514916 build=3738 commit="fa90a986"
INFO [                    main] system info | tid="128766126837760" timestamp=1749514916 n_threads=57 n_threads_batch=-1 total_threads=112 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 8 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 52 key-value pairs and 1147 tensors from /home/mukul/dev-ai/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ4_KS_R4/DeepSeek-R1-0528-IQ4_KS_R4-00001-of-00009.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 0528
llama_model_loader: - kv   3:                            general.version str              = 0528
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek-R1
llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
llama_model_loader: - kv   6:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   7:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   8:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   9:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  10:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  11:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  12:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  13: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  15:                          general.file_type u32              = 345
llama_model_loader: - kv  16:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  17:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  18:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  19:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  20:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  21:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  22:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  23:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  24:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  25:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  26:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  27:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  28:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  29:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  30:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  31: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  32: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  33:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  34:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�..
llama_model_loader: - kv  36:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  37:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  38:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  39:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  40:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  41:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  42:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  43:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  44:               general.quantization_version u32              = 2
llama_model_loader: - kv  45:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/DeepSeek-R1...
llama_model_loader: - kv  46:                   quantize.imatrix.dataset str              = ubergarm-imatrix-calibration-corpus-v...
llama_model_loader: - kv  47:             quantize.imatrix.entries_count i32              = 721
llama_model_loader: - kv  48:              quantize.imatrix.chunks_count i32              = 812
llama_model_loader: - kv  49:                                   split.no u16              = 0
llama_model_loader: - kv  50:                                split.count u16              = 9
llama_model_loader: - kv  51:                        split.tensors.count i32              = 1147
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq4_ks_r4:  116 tensors
llama_model_loader: - type iq5_ks_r4:   58 tensors
llm_load_vocab: special tokens cache size = 818
llm_load_vocab: token to piece cache size = 0.8223 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 129280
llm_load_print_meta: n_merges         = 127741
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 163840
llm_load_print_meta: n_embd           = 7168
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 128
llm_load_print_meta: n_head_kv        = 128
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 24576
llm_load_print_meta: n_embd_v_gqa     = 16384
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 18432
llm_load_print_meta: n_expert         = 256
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 0.025
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = IQ4_KS_R4 - 4.25 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 367.774 GiB (4.701 BPW) 
llm_load_print_meta: repeating layers = 365.940 GiB (4.690 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1 0528
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: LF token         = 131 'Ä'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_layer_dense_lead   = 3
llm_load_print_meta: n_lora_q             = 1536
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 2048
llm_load_print_meta: n_expert_shared      = 1
llm_load_print_meta: expert_weights_scale = 2.5
llm_load_print_meta: expert_weights_norm  = 1
llm_load_print_meta: expert_gating_func   = sigmoid
llm_load_print_meta: rope_yarn_log_mul    = 0.1000
llm_load_tensors: ggml ctx size =    0.93 MiB
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 38317.39 MiB
llm_load_tensors:        CPU buffer size = 42582.45 MiB
llm_load_tensors:        CPU buffer size = 40481.67 MiB
llm_load_tensors:        CPU buffer size = 42840.67 MiB
llm_load_tensors:        CPU buffer size = 40481.67 MiB
llm_load_tensors:        CPU buffer size = 42840.67 MiB
llm_load_tensors:        CPU buffer size = 40481.67 MiB
llm_load_tensors:        CPU buffer size = 42840.67 MiB
llm_load_tensors:        CPU buffer size = 41420.65 MiB
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 17744.02 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 2048
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =  1166.65 MiB
llama_new_context_with_model: KV self size  = 1166.62 MiB, c^KV (q8_0): 1166.62 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =  4252.01 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   312.02 MiB
llama_new_context_with_model: graph nodes  = 8245
llama_new_context_with_model: graph splits = 118
INFO [                    init] initializing slots | tid="128766126837760" timestamp=1749514968 n_slots=1
INFO [                    init] new slot | tid="128766126837760" timestamp=1749514968 id_slot=0 n_ctx_slot=32768
INFO [                    main] model loaded | tid="128766126837760" timestamp=1749514968
INFO [                    main] chat template | tid="128766126837760" timestamp=1749514968 chat_example="You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>" built_in=true
INFO [                    main] HTTP server listening | tid="128766126837760" timestamp=1749514968 n_threads_http="111" port="10002" hostname="0.0.0.0"
INFO [            update_slots] all slots are idle | tid="128766126837760" timestamp=1749514968
```

---

👤 **mtcl** commented the **2025-06-16** at **22:38:57**:<br>

How do i check if my AMX enabled processor is using its "AMX capabilities"? Is there anyway to perform a build on mainline with any specific parameter that enabled AMX and then I can run a comparison between IK and mainline?

---

👤 **ubergarm** commented the **2025-06-16** at **22:56:24**:<br>

@mtcl 

> How do i check if my AMX enabled processor is using its "AMX capabilities"? Is there anyway to perform a build on mainline with any specific parameter that enabled AMX and then I can run a comparison between IK and mainline?

You'd want to see exactly what CPU flags your intel xeon has e.g.

`lscpu | grep -i amx`

On a Intel Xeon 6980P (new enough for AMX extensions) it includes stuff like:
```
amx_bf16 amx_tile amx_int8
```

While mainline llama.cpp does have a few things [as discussed in my intel xeon numa node discussion on mainline with the author commenting in the thread](https://github.com/ggml-org/llama.cpp/issues/12003#issuecomment-2729758792), it didn't yield better performance in my own testing and in general this fork is still faster without any special AMX flags afaict.

You can run comparisons between this fork and mainline using `llama-sweep-bench` which is built in here. I maintain my a branch in my mainline fork on github if you're interested.

I forget your exact rig specs, besides 2x5090s 😛 , but as the linked disussion mentions big models than span multiple NUMA nodes tend to take a hit in performance on AMD and especially Intel rigs depending on BIOS configurations etc.

---

👤 **mtcl** commented the **2025-06-16** at **23:13:18**:<br>

Hey Thank you for the reply!   I checked `lscpu | grep -i amx`, I have all these three these flags `amx_bf16  amx_tile amx_int8`. But how do i make sure that i am compiling the mainline with the correct AMX extensions in the compiled library? Should I use something like this? I do not even know if this is a flag, but i did that anyway `-DGGML_USE_AMX` because I saw it somewhere and cannot locate it anymore. 

```bash
cd ~/dev-ai/llama.cpp

cmake -B build -DGGML_CUDA=ON -DGGML_RPC=ON -DGGML_USE_AMX
cmake --build build --config Release -j 56
```

I was wondering if ik_llama is so awesome without AMX optimizations, how awesome would it be with AMX optimizations!!

And i have 2x5090 + 2x4090s now. I was going to sell 4090s but then i could not lol :)

---

👤 **ubergarm** commented the **2025-06-17** at **00:46:02**:<br>

@mtcl 

> But how do i make sure that i am compiling the mainline with the correct AMX extensions in the compiled library?

Follow [the link in the discussion above where i talk about that on mainline](https://github.com/ikawrakow/ik_llama.cpp/issues/437#issuecomment-2895058400)

>  Should I use something like this?

No

> I was wondering if ik_llama is so awesome without AMX optimizations, how awesome would it be with AMX optimizations!!

It might be slower.

> I was going to sell 4090s but then i could not lol :)
 
112GB VRAM not bad!

---

👤 **mtcl** commented the **2025-06-17** at **00:56:37**:<br>

> > I was going to sell 4090s but then i could not lol :)
>  
> 112GB VRAM not bad!

I would love to hear your thoughts on how to effectively use this much of VRAM. I have started a thread in the discussions and would love to hear your perspective on that.

---

👤 **SlavikCA** commented the **2025-07-06** at **05:01:34**:<br>

I ran this model https://huggingface.co/unsloth/DeepSeek-TNG-R1T2-Chimera-GGUF
with  UD-IQ2_M quants (213 GB) 
Both on llama.cpp (in Docker) and ik_llama.cpp 

System:
- Ubuntu 24.04
- Intel Xeon W5-3425 (12 cores, AMX)
- 512GB DDR5-4800 (8 channels * 64GB), but my memory somehow still not working at the top speed.
- RTX 4090D 48GB VRAM

**llama.cpp:**
```
prompt eval time =   58561.21 ms /  1273 tokens (   46.00 ms per token,    21.74 tokens per second)
       eval time =  371584.74 ms /  1566 tokens (  237.28 ms per token,     4.21 tokens per second)
```

**ik_llama.cpp:**
```
prompt eval time     =   21474.45 ms /  1265 tokens (   16.98 ms per token,    58.91 tokens per second)
generation eval time =  396856.15 ms /  1690 runs   (  234.83 ms per token,     4.26 tokens per second) 
```

So, token generation is about the same, but prompt eval is almost 3x faster on llama.cpp and I think that's because of AMX. But I'm not sure how to confirm that.

llama.cpp params:
```
--model /models/UD-IQ2_M/DeepSeek-TNG-R1T2-Chimera-UD-IQ2_M-00001-of-00005.gguf
--ctx-size 32768
--cache-type-k q8_0 
--cache-type-v q8_0  
--flash-attn
--threads 12
--host 0.0.0.0  --port 37000
--temp 0.6 --top-p 0.95
--n-gpu-layers 999
--override-tensor "blk\.(3|4|5|6|7|8|9|10|11)\.ffn_.*=CUDA0"
--override-tensor exps=CPU
```

llama.cpp logs:
```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no 
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no 
  Device 0: NVIDIA GeForce RTX 4090 D, compute capability 8.9, VMM: yes   
load_backend: loaded CUDA backend from /app/[libggml-cuda.so](http://libggml-cuda.so/)
load_backend: loaded CPU backend from /app/[libggml-cpu-sapphirerapids.so](http://libggml-cpu-sapphirerapids.so/)
build: 5830 (bac8bed2) with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
system_info: n_threads = 12 (n_threads_batch = 12) / 12 
  | CUDA : ARCHS = 500,610,700,750,800,860,890
  | USE_GRAPHS = 1 
  | PEER_MAX_BATCH_SIZE = 128
  | CPU : SSE3 = 1 
  | SSSE3 = 1
  | AVX = 1 
  | AVX2 = 1
  | F16C = 1 
  | FMA = 1
  | BMI2 = 1 
  | AVX512 = 1
  | AVX512_VBMI = 1 
  | AVX512_VNNI = 1
  | AVX512_BF16 = 1 
  | AMX_INT8 = 1
  | LLAMAFILE = 1 
  | OPENMP = 1
  | REPACK = 1
```

ik_llama params:
```
./llama-server \
  --model /models/UD-IQ2_M/DeepSeek-TNG-R1T2-Chimera-UD-IQ2_M-00001-of-00005.gguf \
  --ctx-size 32768 \
  -b 4096 -ub 4096 \
  -ctk q8_0 -fa -mla 3 \
  -amb 512 \
  -fmoe \
  --temp 0.6 --top-p 0.95 \
  --n-gpu-layers 999 \
  --override-tensor "blk\.(3|4|5|6|7|8|9|10)\.ffn_.*=CUDA0" \
  --override-tensor exps=CPU \
  --parallel 1 \
  --threads 12 \
  --host 0.0.0.0 --port 41000
```

ik_llama logs:
```
gml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090 D, compute capability 8.9, VMM: yes
INFO [                    main] build info | tid="123756457574400" timestamp=1751776004 build=3787 commit="0678427f"
INFO [                    main] system info | tid="123756457574400" timestamp=1751776004 n_threads=12 n_threads_batch=-1 total_threads=12 system_info="
| AVX = 1
| AVX_VNNI = 1 
| AVX2 = 1 
| AVX512 = 1 
| AVX512_VBMI = 1 
| AVX512_VNNI = 1 
| AVX512_BF16 = 1 
| FMA = 1 
| NEON = 0 
| SVE = 0 
| ARM_FMA = 0 
| F16C = 1 
| FP16_VA = 0 
| WASM_SIMD = 0 
| BLAS = 1 
| SSE3 = 1 
| SSSE3 = 1 
| VSX = 0 
| MATMUL_INT8 = 0 
| LLAMAFILE = 1 | "
llama_model_loader: (version GGUF V3 (latest))
==========================================================================
Detected incompatible DeepSeek model.
Will try to fix, but there are no guarantees

*** Your prompt processing speed will be crippled ***
```

It's true, that if I use _R4 model - ik_llama will be faster both for PP and TG.
But will it even more faster with AMX?

---

👤 **SlavikCA** commented the **2025-07-06** at **05:01:34**:<br>

I ran this model https://huggingface.co/unsloth/DeepSeek-TNG-R1T2-Chimera-GGUF
with  UD-IQ2_M quants (213 GB) 
Both on llama.cpp (in Docker) and ik_llama.cpp 

System:
- Ubuntu 24.04
- Intel Xeon W5-3425 (12 cores, AMX)
- 512GB DDR5-4800 (8 channels * 64GB), but my memory somehow still not working at the top speed.
- RTX 4090D 48GB VRAM

**llama.cpp:**
```
prompt eval time =   58561.21 ms /  1273 tokens (   46.00 ms per token,    21.74 tokens per second)
       eval time =  371584.74 ms /  1566 tokens (  237.28 ms per token,     4.21 tokens per second)
```

**ik_llama.cpp:**
```
prompt eval time     =   21474.45 ms /  1265 tokens (   16.98 ms per token,    58.91 tokens per second)
generation eval time =  396856.15 ms /  1690 runs   (  234.83 ms per token,     4.26 tokens per second) 
```

So, token generation is about the same, but prompt eval is almost 3x faster on llama.cpp and I think that's because of AMX. But I'm not sure how to confirm that.

llama.cpp params:
```
--model /models/UD-IQ2_M/DeepSeek-TNG-R1T2-Chimera-UD-IQ2_M-00001-of-00005.gguf
--ctx-size 32768
--cache-type-k q8_0 
--cache-type-v q8_0  
--flash-attn
--threads 12
--host 0.0.0.0  --port 37000
--temp 0.6 --top-p 0.95
--n-gpu-layers 999
--override-tensor "blk\.(3|4|5|6|7|8|9|10|11)\.ffn_.*=CUDA0"
--override-tensor exps=CPU
```

llama.cpp logs:
```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no 
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no 
  Device 0: NVIDIA GeForce RTX 4090 D, compute capability 8.9, VMM: yes   
load_backend: loaded CUDA backend from /app/[libggml-cuda.so](http://libggml-cuda.so/)
load_backend: loaded CPU backend from /app/[libggml-cpu-sapphirerapids.so](http://libggml-cpu-sapphirerapids.so/)
build: 5830 (bac8bed2) with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
system_info: n_threads = 12 (n_threads_batch = 12) / 12 
  | CUDA : ARCHS = 500,610,700,750,800,860,890
  | USE_GRAPHS = 1 
  | PEER_MAX_BATCH_SIZE = 128
  | CPU : SSE3 = 1 
  | SSSE3 = 1
  | AVX = 1 
  | AVX2 = 1
  | F16C = 1 
  | FMA = 1
  | BMI2 = 1 
  | AVX512 = 1
  | AVX512_VBMI = 1 
  | AVX512_VNNI = 1
  | AVX512_BF16 = 1 
  | AMX_INT8 = 1
  | LLAMAFILE = 1 
  | OPENMP = 1
  | REPACK = 1
```

ik_llama params:
```
./llama-server \
  --model /models/UD-IQ2_M/DeepSeek-TNG-R1T2-Chimera-UD-IQ2_M-00001-of-00005.gguf \
  --ctx-size 32768 \
  -b 4096 -ub 4096 \
  -ctk q8_0 -fa -mla 3 \
  -amb 512 \
  -fmoe \
  --temp 0.6 --top-p 0.95 \
  --n-gpu-layers 999 \
  --override-tensor "blk\.(3|4|5|6|7|8|9|10)\.ffn_.*=CUDA0" \
  --override-tensor exps=CPU \
  --parallel 1 \
  --threads 12 \
  --host 0.0.0.0 --port 41000
```

ik_llama logs:
```
gml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090 D, compute capability 8.9, VMM: yes
INFO [                    main] build info | tid="123756457574400" timestamp=1751776004 build=3787 commit="0678427f"
INFO [                    main] system info | tid="123756457574400" timestamp=1751776004 n_threads=12 n_threads_batch=-1 total_threads=12 system_info="
| AVX = 1
| AVX_VNNI = 1 
| AVX2 = 1 
| AVX512 = 1 
| AVX512_VBMI = 1 
| AVX512_VNNI = 1 
| AVX512_BF16 = 1 
| FMA = 1 
| NEON = 0 
| SVE = 0 
| ARM_FMA = 0 
| F16C = 1 
| FP16_VA = 0 
| WASM_SIMD = 0 
| BLAS = 1 
| SSE3 = 1 
| SSSE3 = 1 
| VSX = 0 
| MATMUL_INT8 = 0 
| LLAMAFILE = 1 | "
llama_model_loader: additional 4 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 69 key-value pairs and 1086 tensors from /home/slavik/.cache/huggingface/hub/models--unsloth--DeepSeek-TNG-R1T2-Chimera-GGUF/snapshots/1703b3d3bc20c493045ecc8e521a12a62d3b83a6/UD-IQ2_M/DeepSeek-TNG-R1T2-Chimera-UD-IQ2_M-00001-of-00005.gguf (version GGUF V3 (latest))
==========================================================================
Detected incompatible DeepSeek model.
Will try to fix, but there are no guarantees

*** Your prompt processing speed will be crippled ***
```

---

👤 **ikawrakow** commented the **2025-07-06** at **05:24:45**:<br>

> So, token generation is about the same, but prompt eval is almost 3x faster on llama.cpp and I think that's because of AMX. But I'm not sure how to confirm that.

You mean `ik_llama.cpp` is almost 3X faster than `llama.cpp`?

---

👤 **SlavikCA** commented the **2025-07-06** at **05:29:39**:<br>

🤦 
You're right.
I need to go to sleep for a little bit.