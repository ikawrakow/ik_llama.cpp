### 🐛 [#523](https://github.com/ikawrakow/ik_llama.cpp/issues/523) - Bug: tg speed drop after https://github.com/ikawrakow/ik_llama.cpp/pull/518

| **Author** | `ciprianveg` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-12 |
| **Updated** | 2025-06-13 |

---

#### Description

### What happened?

 tg speed drop after https://github.com/ikawrakow/ik_llama.cpp/pull/518 to 4.5 t/s from 5.5t/s after https://github.com/ikawrakow/ik_llama.cpp/pull/517 on deepseek r1 iQ3XXS UD. This is when I do not use -rtr. If I use -rtr, pp speed drops from 250t/s to 26t/s also before and also after  https://github.com/ikawrakow/ik_llama.cpp/pull/518:
./build/bin/llama-sweep-bench   \
    --model /media/ciprian/m2/ai/models/Deepseek-R1-2805-Q3-XXS-UD/DeepSeek-R1-0528-UD-IQ3_XXS-00001-of-00006.gguf \
    --alias DeepSeek-R1-0528-UD-IQ3_XXS \
    --ctx-size 71680 \
    -ctk q8_0 \
    -mla 3 \
    -fa \
    -amb 256 \
    -fmoe -rtr \
    --temp 0.5 \
    --top_p 0.95 \
    --min_p 0.01 \
    --n-gpu-layers 63 \
    -ot "blk\.[0-4]\.ffn_up_exps=CUDA0,blk\.[0-4]\.ffn_gate_exps=CUDA0,blk\.[0-2]\.ffn_down_exps=CUDA0"  \
    -ot "blk\.1[0-2]\.ffn_up_exps=CUDA1,blk\.1[0-2]\.ffn_gate_exps=CUDA1"    \
    -ot "blk\.1[3-4]\.ffn_up_exps=CUDA2,blk\.1[3-4]\.ffn_gate_exps=CUDA2"    \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 16 \
    --threads-batch 16 \
    --host 0.0.0.0 --port 5002   \
    --ubatch-size 7168 --batch-size 7168  --no-mmap 

### Name and Version

llama-server, ubuntu, TR 3955wx, 256GB ddr4, 3x3090

cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF  -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_IQK_FORCE_BF16=1 -DGGML_CCACHE=OFF


### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-06-12** at **07:50:42**:<br>

So, what is it: is the TG speed drop for `IQ3_S` or for `IQ3_XXS`? Or for both? (but there is only one performance value given).

On the two systems I have available (Zen3 and Zen4), TG performance is exactly the same as before (and I don't see a reason why it should decrease by 20%). Have you tried dropping caches?

The reason you see a low PP performance when you use `-rtr` with these models is that there is no CUDA implementation for `IQ3_S_R4` or `IQ3_XXS_R4`, so the matrix multiplications for the experts left in RAM is done on the CPU, and your CPU seems to be on the low-end performance side (people do get over 100 t/s on high-end CPUs running CPU-only). So, the only case where you would want to use `-rtr` with a quant that does not have a CUDA implementation for the interleaved variant is when your prompts are relatively short, so offloading to the GPU is slower than running on the CPU. But after PRs #516 and #518, normally prompt processing should now be faster without `-rtr` for `IQ3_S` and `IQ3_XXS`.

---

👤 **ikawrakow** commented the **2025-06-12** at **08:00:48**:<br>

> and your CPU seems to be on the low-end performance side

Take that back. You have decided to use the quants with the lowest CPU performance (`IQ3_S` and `IQ3_XXS`), so 25 t/s for DeepSeek-R1 with these quants is not too bad. PP should be 3X better after PR #516 and #518 when running on the CPU.

---

👤 **ciprianveg** commented the **2025-06-12** at **08:34:02**:<br>

Hi, sorry if I was not clear:
Using DeepSeek-R1-0528-UD-IQ3_XXS,
After https://github.com/ikawrakow/ik_llama.cpp/pull/517 tg speed increased a little to 5.5t/s (without -rtr and 6.4 with rtr). 
After https://github.com/ikawrakow/ik_llama.cpp/pull/518  tg speed drop to 4.5 t/s (without -rtr and 6.2 with rtr). 

If I use -rtr, even after pr 516, 518, pp speed drops from cca 250t/s to 26t/s.

I realize that quant is not a good fit, but i tried it because is the biggest one I can fit on my ram+vram, I wanted something a little bigger and possibly better perplexity wise than the already good and fast Ubergram's IQ2_K_R4 model..

---

👤 **ciprianveg** commented the **2025-06-12** at **08:34:02**:<br>

Hi, sorry if I was not clear:
Using DeepSeek-R1-0528-UD-IQ3_XXS,
After https://github.com/ikawrakow/ik_llama.cpp/pull/517 tg speed was 5.5t/s (without -rtr and 6.4 with rtr). 
After https://github.com/ikawrakow/ik_llama.cpp/pull/518  tg speed drop to 4.5 t/s (without -rtr and 6.2 with rtr). 

If I use -rtr, even after pr 516, 518, pp speed drops from cca 250t/s to 26t/s.

I realize that quant is not a good fit, but i tried it because is the biggest one I can fit on my ram+vram, I wanted something a little bigger and possibly better perplexity wise than the already good and fast Ubergram's IQ2_K_R4 model..

---

👤 **ikawrakow** commented the **2025-06-12** at **11:06:45**:<br>

So, after #517 it became slightly faster. Which means that what I did in #516 for `IQ3_XXS` is slightly better on your system. But after #518, which applies the very same approach used in #516 to `IQ3_S`, it suddenly became 20% slower. Looking at the Unsloth `IQ3_XXS` model, I see they have used `IQ3_S` for the routed experts in 5 layers. I.e., less than 10% of the computation is done according to the new approach of #518. In order to observe a 20% drop in performance, simple napkin math tells me that `IQ3_S` GEMV must have become 3 times slower with PR #518. Sorry, but this seems extremely unlikely.

You didn't try to drop caches as suggested, did you?
```
echo 3 | sudo tee /proc/sys/vm/drop_caches
```

---

👤 **Ph0rk0z** commented the **2025-06-12** at **11:23:18**:<br>

I observe a similar thing:

/DeepSeek-R1-0528-UD-IQ1_S-00001-of-00004.gguf

Pre changes

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   25.191 |   162.60 |  102.925 |     9.95 |
|  4096 |   1024 |   4096 |   26.593 |   154.02 |  105.827 |     9.68 |
|  4096 |   1024 |   8192 |   28.833 |   142.06 |  110.305 |     9.28 |


All changes

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   24.955 |   164.13 |  104.894 |     9.76 |
|  4096 |   1024 |   4096 |   26.257 |   156.00 |  107.417 |     9.53 |
|  4096 |   1024 |   8192 |   28.061 |   145.97 |  111.293 |     9.20 |


Up to #517

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   24.214 |   169.16 |  100.856 |    10.15 |
|  4096 |   1024 |   4096 |   25.692 |   159.43 |  104.756 |     9.78 |
|  4096 |   1024 |   8192 |   27.709 |   147.82 |  108.117 |     9.47 |

I have 2 copies of the repo so I can test head to head.

An R4 quant is nonviable since it drops PP down to 50/60 unless using batch-ubatch 2048/1024 as I benchmark. Assuming same thing happens in R4 as using RTR flag.

---

👤 **ikawrakow** commented the **2025-06-12** at **11:29:39**:<br>

In what sense is a <2% change similar to a 20% change?

---

👤 **Ph0rk0z** commented the **2025-06-12** at **11:43:54**:<br>

It confirms there is a change at all. On his particular hardware maybe the change is larger.

---

👤 **ciprianveg** commented the **2025-06-12** at **11:46:09**:<br>

> So, after [#517](https://github.com/ikawrakow/ik_llama.cpp/pull/517) it became slightly faster. Which means that what I did in [#516](https://github.com/ikawrakow/ik_llama.cpp/pull/516) for `IQ3_XXS` is slightly better on your system. But after [#518](https://github.com/ikawrakow/ik_llama.cpp/pull/518), which applies the very same approach used in [#516](https://github.com/ikawrakow/ik_llama.cpp/pull/516) to `IQ3_S`, it suddenly became 20% slower. Looking at the Unsloth `IQ3_XXS` model, I see they have used `IQ3_S` for the routed experts in 5 layers. I.e., less than 10% of the computation is done according to the new approach of [#518](https://github.com/ikawrakow/ik_llama.cpp/pull/518). In order to observe a 20% drop in performance, simple napkin math tells me that `IQ3_S` GEMV must have become 3 times slower with PR [#518](https://github.com/ikawrakow/ik_llama.cpp/pull/518). Sorry, but this seems extremely unlikely.
> 
> You didn't try to drop caches as suggested, did you?
> 
> ```
> echo 3 | sudo tee /proc/sys/vm/drop_caches
> ```

i will execute echo 3 | sudo tee /proc/sys/vm/drop_caches, and rerun the test on main, rebuild than on 517, re-drop the cache and rerun it, and i will report back with the results, thank you and sorry if this will be only a cache related issue on my side

---

👤 **ikawrakow** commented the **2025-06-12** at **12:08:01**:<br>

> It confirms there is a change at all. On his particular hardware maybe the change is larger.

Does it? The fluctuations in performance I observe from run to run are definitely larger than 2%. `llama-sweep-bench`, unlike `llama-bench`,  does a single run for each `N_KV`. Your system must be very different from any other system I have seen if performance stays within better than 2% from one run to another. If you ran it 10 times, computed the average and the standard deviation, and then we saw that the difference is larger than 3 standard deviations, then we would know that performance really changed.

---

👤 **Ph0rk0z** commented the **2025-06-12** at **12:24:14**:<br>

Dunno, there is some variance for sure. I've run many of them. The all changes drop does look like a drop tho. They tend to be repeatable when you have the same settings, especially on the initial/final runs. When you add/remove layers or change settings is when it gets dicey. It smells like with that middle one, I'll never see 10s on TG anymore. Lets see what he comes back with.

---

👤 **ciprianveg** commented the **2025-06-12** at **13:31:49**:<br>

very strange, i redone the tests dropping the cache after clean rebuild and the difference is big, but the difference is big comparing to origin/ik/iq1_s_gemm. Before #516 and #517 I had a smaller 4.5 t/s tg speed so also something good happened yesterday. 
I assume I should not be choosing the worst type of quant for ik_llama (DeepSeekR1-UD-iQ3-XXS), so I switched back to ubergam's q2 and wait nicely till him or someone else can make a slightly bigger. ik compatible quant and enjoy 8t/s tg speed and same 250 t/s pp speed:)

My tests:
Step 1) delete build directory and build from ik/iq1_s_gemm with -DGGML_CCACHE=OFF
git checkout -b iq1_s_gemm origin/ik/iq1_s_gemm
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF  -DGGML_SCHED_MAX_COPIES=1  -DGGML_CUDA_IQK_FORCE_BF16=1 -DGGML_CCACHE=OFF
cmake --build ./build --config Release -j $(nproc)

echo 3 | sudo tee /proc/sys/vm/drop_caches

./startDeepSeekR1-UD-iQ3-XXS.sh  (the llama-sweep-bench command from my 1st post)

main: n_kv_max = 71680, n_batch = 7168, n_ubatch = 7168, flash_attn = 1, n_gpu_layers = 63, n_threads = 16, n_threads_batch = 16

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  7168 |   1792 |      0 |   28.054 |   255.50 |  322.494 |     5.56 |

step 2) delete build dir, and rebuild from main:
 2011  git checkout main
 2012  git pull
 2013  cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF  -DGGML_SCHED_MAX_COPIES=1  -DGGML_CUDA_IQK_FORCE_BF16=1 -DGGML_CCACHE=OFF
 2014  cmake --build ./build --config Release -j $(nproc)
 2015  echo 3 | sudo tee /proc/sys/vm/drop_caches
 2016  history

./startDeepSeekR1-UD-iQ3-XXS.sh  

main: n_kv_max = 71680, n_batch = 7168, n_ubatch = 7168, flash_attn = 1, n_gpu_layers = 63, n_threads = 16, n_threads_batch = 16

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  7168 |   1792 |      0 |   28.007 |   255.94 |  414.487 |     4.32 |
|  7168 |   1792 |   7168 |   36.711 |   195.25 |  428.074 |     4.19 |



System: TR 3955WX 256GB ram, 2x3090 24GB + A4500 20GB

---

👤 **ikawrakow** commented the **2025-06-12** at **16:17:20**:<br>

Can you try #524 ?

My guess is we are running into compiler limitations. The matrix multiplication code uses C++ templates, and I have observed in the past the strange effect that after adding a new instantiation of the template, performance suddenly drops for pre-existing template instances. I haven't seen this effect for a while, but maybe it is there for you?

What is the compiler version you are using?

---

👤 **ciprianveg** commented the **2025-06-12** at **16:22:07**:<br>

I will try in about 2h and let you know. Thank you!

---

👤 **ciprianveg** commented the **2025-06-12** at **19:10:24**:<br>

hello, much better: origin/ik/iq_gemv_tweaks :)
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  7168 |   1792 |      0 |   28.049 |   255.56 |  329.723 |     5.43 |

vs previous test on main:
PP	TG	N_KV	T_PP s	S_PP t/s	T_TG s	S_TG t/s
7168	1792	0	28.007	255.94	414.487	4.32