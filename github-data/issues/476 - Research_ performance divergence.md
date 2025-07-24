### 📝 [#476](https://github.com/ikawrakow/ik_llama.cpp/issues/476) - Research: performance divergence

| **Author** | `VinnyG9` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-05-30 |
| **Updated** | 2025-06-14 |

---

#### Description

### Research Stage

- [ ] Background Research (Let's try to avoid reinventing the wheel)
- [ ] Hypothesis Formed (How do you think this will work and it's effect?)
- [ ] Strategy / Implementation Forming
- [x] Analysis of results
- [x] Debrief / Documentation (So people in the future can learn from us)

### Previous existing literature and research

when i ran benches previously i got pretty good results on cpu inference like 30-40t/s on qwen3 30B, now i am trying to run the server for aider and the speed is less than half is it expected??

### Hypothesis

_No response_

### Implementation

_No response_

### Analysis

_No response_

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-31** at **05:22:24**:<br>

Please be specific in your issue. Provide quantization type used, system information, full command line to start the server and, ideally, last good/first bad commit where you observe the performance change. See #474 for an example.

---

👤 **VinnyG9** commented the **2025-05-31** at **10:18:09**:<br>

I've been testing ik_llama.cpp for about a month mostly benchmarks can't report any regression
 running bare build time flags(NATIVE=1, CUDA=1, CUDA_ARCH) and runtime flags(rtr, fa, fmoe, numa)

latest ubuntu/mint

no matter the model i try, dense, MoE etc i get less than 50% performance than the benchmarks show, when running mainline the benchmark numbers are way lower but are consistent with performance numbers when running the server

---

👤 **ikawrakow** commented the **2025-05-31** at **14:28:49**:<br>

So, the issue is that the performance you observe when running `llama-server` is 2X lower than the performance you observe when running one of the benchmark tools?

Is the PP performance affected, or the TG performance, or both?

Generic statement will lead nowhere (other than the issue getting closed)

---

👤 **VinnyG9** commented the **2025-06-01** at **03:16:25**:<br>

> So, the issue is that the performance you observe when running `llama-server` is 2X lower than the performance you observe when running one of the benchmark tools?

yes 

> Is the PP performance affected, or the TG performance, or both?
> 

both, literally a bit less than half PP/TG. think it could be a numa issue? i tried with stock bios settings but got worse results albeit closer bench/serve numbers

---

👤 **saood06** commented the **2025-06-01** at **03:26:18**:<br>

> both, literally a bit less than half PP/TG. think it could be a numa issue? i tried with stock bios settings but got worse results albeit closer bench/serve numbers

Please provide the exact commands used for benching and for server. 

I brought `llama-sweep-bench` over to this repo and use it regularly because in my experience it does accurately reflect server performance (including how it changes across different depths), to the point where I run it to validate that the model has warmed up and loaded into RAM correctly (as my server is very sensitive to memory placement and the model is stored on HDDs so performance is unusable until the model is warmed up).

---

👤 **Ph0rk0z** commented the **2025-06-01** at **13:24:16**:<br>

Its funny because I often get slightly better speeds on server than the sweep bench. Nowhere near half so something is wrong.

Only numa thing that helps is adding interleave=all to the command line you run. Setting load balancing in the kernel to 0 doesn't do move the needle one way or another despite the warning.

One thing I did notice is that the bench can be a little irregular at times by a t/s here or there. May I also suggest getting ccmake when compiling so you can set your flags and forget it. 

edit:

So playing with 4096 batches showed me something. In server, prompt speed on smaller prompts prints as half speed or less.I was getting 110 max and on a 2k token prompt would hit 60-70 reported. A large enough prompt still returns the correct speed. Can't explain 1/2 TG tho.

---

👤 **VinnyG9** commented the **2025-06-02** at **16:34:43**:<br>

> > both, literally a bit less than half PP/TG. think it could be a numa issue? i tried with stock bios settings but got worse results albeit closer bench/serve numbers
> 
> Please provide the exact commands used for benching and for server.
> 
> I brought `llama-sweep-bench` over to this repo and use it regularly because in my experience it does accurately reflect server performance (including how it changes across different depths), to the point where I run it to validate that the model has warmed up and loaded into RAM correctly (as my server is very sensitive to memory placement and the model is stored on HDDs so performance is unusable until the model is warmed up).

# server:
```
  "Qwen3-30B-MoE":
    env:
      - "CUDA_VISIBLE_DEVICES= "
    proxy: "http://192.168.15.101:9999"
    cmd: |
      /ssd/share/Software/backends/ik_llama.cpp/build/bin/llama-server
      --host localhost --port 9999 --flash-attn
      --cache-type-k f16 --cache-type-v f16
      --ctx-size 40960
      --samplers "top_k;top_p;min_p;temperature;typ_p;xtc"
      --temp 0.6 --repeat-penalty 1.0
      --min-p 0.01 --top-k 20 --top-p 0.95
      -ngl 0 -rtr -fmoe -ser 7,1 --threads 31 --numa distribute
      --model /models/gguf/MoE/Qwen3-30B-A3B-128K-UD-Q4_K_XL.gguf
```
## output
```
INFO [   launch_slot_with_task] slot is processing task | tid="123321346306048" timestamp=1748881719 id_slot=0 id_task=399
INFO [            update_slots] kv cache rm [p0, end) | tid="123321346306048" timestamp=1748881719 id_slot=0 id_task=399 p0=450
INFO [           print_timings] prompt eval time     =    3133.99 ms /   388 tokens (    8.08 ms per token,   123.80 tokens per second) | tid="123321346306048" timestamp=1748881752 id_slot=0 id_task=399 t_prompt_processing=3133.991 n_prompt_tokens_processed=388 t_token=8.077296391752578 n_tokens_second=123.80380160632241
INFO [           print_timings] generation eval time =   29634.35 ms /   400 runs   (   74.09 ms per token,    13.50 tokens per second) | tid="123321346306048" timestamp=1748881752 id_slot=0 id_task=399 t_token_generation=29634.354 n_decoded=400 t_token=74.085885 n_tokens_second=13.497847801912604
INFO [           print_timings]           total time =   32768.35 ms | tid="123321346306048" timestamp=1748881752 id_slot=0 id_task=399 t_prompt_processing=3133.991 t_token_generation=29634.354 t_total=32768.345
INFO [            update_slots] slot released | tid="123321346306048" timestamp=1748881752 id_slot=0 id_task=399 n_ctx=40960 n_past=1237 n_system_tokens=0 n_cache_tokens=1237 truncated=false
INFO [            update_slots] all slots are idle | tid="123321346306048" timestamp=1748881752

```

# sweep-bench:
`CUDA_VISIBLE_DEVICES= numactl --interleave=all /ssd/share/Software/backends/ik_llama.cpp/build/bin/llama-sweep-bench -m /models/gguf/MoE/Qwen3-30B-A3B-128K-UD-Q4_K_XL.gguf -rtr -fa -fmoe --numa distribute -t 31 -c 8196 -b 2048 -ub 512`

```
main: n_kv_max = 8448, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = -1, n_threads = 31, n_threads_batch = 31

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.981 |   258.43 |    4.757 |    26.91 |
|   512 |    128 |    512 |    2.157 |   237.35 |    4.642 |    27.57 |
|   512 |    128 |   1024 |    2.421 |   211.47 |    5.040 |    25.40 |
|   512 |    128 |   1536 |    2.844 |   180.04 |    4.951 |    25.85 |
|   512 |    128 |   2048 |    2.991 |   171.20 |    5.313 |    24.09 |
|   512 |    128 |   2560 |    3.222 |   158.89 |    5.136 |    24.92 |
|   512 |    128 |   3072 |    3.525 |   145.24 |    5.442 |    23.52 |
|   512 |    128 |   3584 |    3.758 |   136.25 |    5.559 |    23.03 |
|   512 |    128 |   4096 |    4.089 |   125.20 |    5.580 |    22.94 |
|   512 |    128 |   4608 |    4.262 |   120.14 |    5.563 |    23.01 |
|   512 |    128 |   5120 |    4.832 |   105.96 |    6.061 |    21.12 |
|   512 |    128 |   5632 |    4.954 |   103.36 |    6.060 |    21.12 |
|   512 |    128 |   6144 |    5.218 |    98.12 |    6.202 |    20.64 |
|   512 |    128 |   6656 |    5.664 |    90.40 |    6.193 |    20.67 |
|   512 |    128 |   7168 |    5.776 |    88.65 |    6.122 |    20.91 |
|   512 |    128 |   7680 |    6.135 |    83.46 |    7.535 |    16.99 |
failed to decode the batch, n_batch = 2048, ret = 1
main: llama_decode() failed
```
# llama-bench

```
CUDA_VISIBLE_DEVICES= /ssd/share/Software/backends/ik_llama.cpp/build/bin/llama-bench -m /models/gguf/MoE/Qwen3-30B-A3B-128K-UD-Q4_K_XL.gguf -t 31 --numa distribute -rtr 1 -fa 1 -fmoe 1 -n 64,128,256,512 -p 512,1024,2048,4096
ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected
WARNING: /proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance
```

| model                          |       size |     params | backend    | ngl | threads | fa | rtr | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -: | --: | ---: | ------------: | ---------------: |
============ Repacked 337 tensors
| qwen3moe ?B Q4_K - Medium      |  16.49 GiB |    30.53 B | CUDA       |  99 |      31 |  1 |   1 |    1 |         pp512 |    254.35 ± 7.37 |
| qwen3moe ?B Q4_K - Medium      |  16.49 GiB |    30.53 B | CUDA       |  99 |      31 |  1 |   1 |    1 |        pp1024 |    226.91 ± 7.88 |
| qwen3moe ?B Q4_K - Medium      |  16.49 GiB |    30.53 B | CUDA       |  99 |      31 |  1 |   1 |    1 |        pp2048 |    206.85 ± 6.65 |
| qwen3moe ?B Q4_K - Medium      |  16.49 GiB |    30.53 B | CUDA       |  99 |      31 |  1 |   1 |    1 |        pp4096 |    163.43 ± 2.74 |
| qwen3moe ?B Q4_K - Medium      |  16.49 GiB |    30.53 B | CUDA       |  99 |      31 |  1 |   1 |    1 |          tg64 |     32.71 ± 0.89 |
| qwen3moe ?B Q4_K - Medium      |  16.49 GiB |    30.53 B | CUDA       |  99 |      31 |  1 |   1 |    1 |         tg128 |     33.11 ± 0.54 |
| qwen3moe ?B Q4_K - Medium      |  16.49 GiB |    30.53 B | CUDA       |  99 |      31 |  1 |   1 |    1 |         tg256 |     31.07 ± 1.80 |
| qwen3moe ?B Q4_K - Medium      |  16.49 GiB |    30.53 B | CUDA       |  99 |      31 |  1 |   1 |    1 |         tg512 |     27.44 ± 3.13 |

---

👤 **VinnyG9** commented the **2025-06-02** at **16:55:21**:<br>

and running on GPU

```
INFO [            update_slots] all slots are idle | tid="123510069428224" timestamp=1748883135
INFO [   launch_slot_with_task] slot is processing task | tid="123510069428224" timestamp=1748883146 id_slot=0 id_task=408
INFO [            update_slots] kv cache rm [p0, end) | tid="123510069428224" timestamp=1748883146 id_slot=0 id_task=408 p0=456
INFO [           print_timings] prompt eval time     =    1171.24 ms /   381 tokens (    3.07 ms per token,   325.30 tokens per second) | tid="123510069428224" timestamp=1748883151 id_slot=0 id_task=408 t_prompt_processing=1171.238 n_prompt_tokens_processed=381 t_token=3.0741154855643047 n_tokens_second=325.2968226782259
INFO [           print_timings] generation eval time =    3364.22 ms /   124 runs   (   27.13 ms per token,    36.86 tokens per second) | tid="123510069428224" timestamp=1748883151 id_slot=0 id_task=408 t_token_generation=3364.215 n_decoded=124 t_token=27.13076612903226 n_tokens_second=36.8585242025257
INFO [           print_timings]           total time =    4535.45 ms | tid="123510069428224" timestamp=1748883151 id_slot=0 id_task=408 t_prompt_processing=1171.238 t_token_generation=3364.215 t_total=4535.453
INFO [            update_slots] slot released | tid="123510069428224" timestamp=1748883151 id_slot=0 id_task=408 n_ctx=40960 n_past=960 n_system_tokens=0 n_cache_tokens=960 truncated=false

```

main: n_kv_max = 8448, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 31, n_threads_batch = 31

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.241 |   412.49 |    2.392 |    53.51 |
|   512 |    128 |    512 |    1.186 |   431.64 |    2.512 |    50.96 |
|   512 |    128 |   1024 |    1.223 |   418.59 |    2.613 |    48.99 |
|   512 |    128 |   1536 |    1.232 |   415.57 |    2.713 |    47.18 |
|   512 |    128 |   2048 |    1.275 |   401.43 |    2.819 |    45.41 |
|   512 |    128 |   2560 |    1.267 |   403.95 |    2.933 |    43.64 |
|   512 |    128 |   3072 |    1.318 |   388.40 |    3.042 |    42.07 |
|   512 |    128 |   3584 |    1.332 |   384.52 |    3.146 |    40.68 |
|   512 |    128 |   4096 |    1.366 |   374.89 |    3.267 |    39.18 |
|   512 |    128 |   4608 |    1.377 |   371.86 |    3.386 |    37.80 |
|   512 |    128 |   5120 |    1.389 |   368.53 |    3.533 |    36.23 |
|   512 |    128 |   5632 |    1.409 |   363.27 |    3.633 |    35.23 |
|   512 |    128 |   6144 |    1.432 |   357.60 |    3.710 |    34.51 |
|   512 |    128 |   6656 |    1.458 |   351.20 |    3.796 |    33.72 |
|   512 |    128 |   7168 |    1.492 |   343.10 |    3.905 |    32.78 |
|   512 |    128 |   7680 |    1.493 |   342.86 |    4.017 |    31.86 |
failed to decode the batch, n_batch = 2048, ret = 1
main: llama_decode() failed

---

👤 **saood06** commented the **2025-06-03** at **00:42:25**:<br>

Is there any reason why you use 31 threads? I would say try using 32 threads and see if that helps your performance (but I don't think that is the reason for the gap in performance between server and sweep).

See this comment about why that might be a bad choice: https://github.com/ikawrakow/ik_llama.cpp/discussions/223#discussioncomment-12292591

---

👤 **VinnyG9** commented the **2025-06-03** at **01:37:17**:<br>

> Is there any reason why you use 31 threads? I would say try using 32 threads and see if that helps your performance (but I don't think that is the reason for the gap in performance between server and sweep).
> 
> See this comment about why that might be a bad choice: [#223 (comment)](https://github.com/ikawrakow/ik_llama.cpp/discussions/223#discussioncomment-12292591)

yeah when i benched it performance improved with the number of(physical) threads up to 31-32, though only for the moe's 


is it normal that during generation the model pauses on every comma? i find it funny

---

👤 **nux** commented the **2025-06-03** at **02:28:40**:<br>

Not sure if relevant here - the topic name seems so. Was looking into some performance issues and found this thread.

nux@red ~/dev/ik_llama.cpp $ ./build/bin/llama-bench -m /mnt/nvme/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4/DeepSeek-V3-0324-IQ4_K_R4-00001-of-00010.gguf -p 512 -t 32 -mla 2 -fa 1 -fmoe 1 -ngl 99 --override-tensor "exps=CPU" -amb 512
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
| model                          |       size |     params | backend    | ngl | fa | mla |   amb | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --: | ----: | ---: | ------------: | ---------------: |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CUDA       |  99 |  1 |   2 |   512 |    1 |         pp512 |     28.36 ± 0.03 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CUDA       |  99 |  1 |   2 |   512 |    1 |         tg128 |      4.80 ± 0.01 |

I had a .txt file of an older benchmark showing this:
nux@red ~/dev/ik_llama.cpp $ ./build/bin/llama-bench -m /mnt/nvme/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4/DeepSeek-V3-0324-IQ4_K_R4-00001-of-00010.gguf -p 512 -t 32 -mla 2 -fa 1 -fmoe 1 -ngl 99 --override-tensor "exps=CPU" -amb 512
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
| model                          |       size |     params | backend    | ngl | fa | mla |   amb | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --: | ----: | ---: | ------------: | ---------------: |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CUDA       |  99 |  1 |   2 |   512 |    1 |         pp512 |     76.13 ± 2.43 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CUDA       |  99 |  1 |   2 |   512 |    1 |         tg128 |      9.79 ± 0.08 |

build: 1ea1df4b (3659)

I checked out the same checkout and ran it again and results were the same.

Am I missing anything obvious here? Something change that I need to adjust for?

Building like this:
cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build build --config Release -j --clean-first


Running on 2x9115, 768gb ram, 3090 gpu

I ended up troubleshooting the performance I am seeing now when trying to figure out why ubergarm/DeepSeek-R1-0528-GGUF/IQ4_KS_R4 was running slowly.

If you think it makes sense for me to open a new issue I can, as my tg/pp have both slowed down, unlike what I'm reading about above

Thanks

---

👤 **ikawrakow** commented the **2025-06-03** at **05:06:02**:<br>

@Fuckingnameless 

Your system seems to be one of those that are extremely finicky about tensor placement in RAM. Looking at the `llama-bench` vs `llama-sweep-bench` TG results I see a 20% difference in TG performance at zero context. There is an obvious difference between your server command and your benchmark runs: context is 41k tokens for the server and 8k or less in the benchmarks. This potentially changes where things go in RAM. Also, seeing `numactl` and `--numa` involved immediately rises red flags. Do you have a dual socket system? (I don't remember the system configurations of all users, so adding the details of your system to the issue would be useful).

Having said all this, I still find a factor of 2 difference in CPU performance strange. The difference is much less on CUDA, so I would focus on trying to resolve the CPU performance first.

---

👤 **ikawrakow** commented the **2025-06-03** at **05:20:03**:<br>

@nux 

There was PR #461 that added CUDA implementation for some of the row-interleaved quants. This results in a change in behavior for your `IQ4_K_R4` quantized model: prior to PR #461 all matrix multiplications for `X_R4` tensors had to be done on the CPU. After PR #461, for batch size `>= 32` they get offloaded to the GPU to perform the matrix multiplications. If the PCI-E speed is low for some reason, this can make PP slower. You can try adding `-op 26,0,27,0,29,0` to the command line to see what happens. This will disable the offload to the GPU.

I have no explanation for the 2X lower TG performance. Try using `-mla 3`, which has been supported on the GPU since PR #408/#413

---

👤 **nux** commented the **2025-06-03** at **12:56:00**:<br>

I will put together a script to go through commits and benchmark to figure out exactly when this started. I'm noticing right now is that while llama-bench is running, the GPU utilization drops to 38-39% for about 10 seconds and going back up to 99%. While  llama-bench is running I see this pattern repeating with gpu usage %

I have been using mla 3 - but ran the benchmark above in mla 2 for comparison purposes. PCI-E is 16x. Will post when I figure out what commit performance went down

---

👤 **nux** commented the **2025-06-03** at **14:00:55**:<br>

Commit 0976467 is when the performance went down for me. Was running for i in `cut -d " " -f1 commits.txt `;do git checkout $i;./cmd-build.sh ;./start-bench.sh >> results.txt;done

start-bench is: ./build/bin/llama-bench -m /mnt/nvme/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4/DeepSeek-V3-0324-IQ4_K_R4-00001-of-00010.gguf -p 512 -t 32 -mla 2 -fa 1 -fmoe 1 -ngl 99 --override-tensor "exps=CPU" -amb 512

build: ccd6d9cd (3716)
| model                          |       size |     params | backend    | ngl | fa | mla |   amb | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --: | ----: | ---: | ------------: | ---------------: |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CUDA       |  99 |  1 |   2 |   512 |    1 |         pp512 |     26.74 ± 0.05 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CUDA       |  99 |  1 |   2 |   512 |    1 |         tg128 |      4.80 ± 0.00 |

build: 09764678 (3715)
| model                          |       size |     params | backend    | ngl | fa | mla |   amb | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --: | ----: | ---: | ------------: | ---------------: |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CUDA       |  99 |  1 |   2 |   512 |    1 |         pp512 |     26.75 ± 0.04 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CUDA       |  99 |  1 |   2 |   512 |    1 |         tg128 |      4.81 ± 0.00 |

build: 14292913 (3714)
| model                          |       size |     params | backend    | ngl | fa | mla |   amb | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --: | ----: | ---: | ------------: | ---------------: |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CUDA       |  99 |  1 |   2 |   512 |    1 |         pp512 |     76.24 ± 1.44 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CUDA       |  99 |  1 |   2 |   512 |    1 |         tg128 |     10.08 ± 0.06 |

build: 24c010b3 (3713)
| model                          |       size |     params | backend    | ngl | fa | mla |   amb | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --: | ----: | ---: | ------------: | ---------------: |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CUDA       |  99 |  1 |   2 |   512 |    1 |         pp512 |     77.25 ± 0.70 |
| deepseek2 671B IQ4_K_R4 - 4.5 bpw | 386.18 GiB |   672.05 B | CUDA       |  99 |  1 |   2 |   512 |    1 |         tg128 |     10.07 ± 0.06 |

---

👤 **ikawrakow** commented the **2025-06-03** at **14:10:23**:<br>

@nux Maybe it is better you open a new issue with your findings. You can also add the tensors being used in your model when you do so. This issue is about a discrepancy between performance observed with `llama-bench`/`llama-sweep-bench` and performance observed with `llama-server`.

---

👤 **VinnyG9** commented the **2025-06-10** at **00:38:42**:<br>

> @Fuckingnameless
> 
> Your system seems to be one of those that are extremely finicky about tensor placement in RAM. Looking at the `llama-bench` vs `llama-sweep-bench` TG results I see a 20% difference in TG performance at zero context. There is an obvious difference between your server command and your benchmark runs: context is 41k tokens for the server and 8k or less in the benchmarks. This potentially changes where things go in RAM. Also, seeing `numactl` and `--numa` involved immediately rises red flags. Do you have a dual socket system? (I don't remember the system configurations of all users, so adding the details of your system to the issue would be useful).
> 
> Having said all this, I still find a factor of 2 difference in CPU performance strange. The difference is much less on CUDA, so I would focus on trying to resolve the CPU performance first.

it happens on qwen3 30b but not the 235b moe in which i see almost 1:1 numbers for sweep-bench vs API, also ran a dense 70b test and TG was 1:1


i make sure the runtime flags are equal between runs, should i be building with:
-DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=ON
?

---

👤 **cg10036** commented the **2025-06-14** at **15:58:34**:<br>

Hi, I'm leaving a comment because I seem to be experiencing a similar issue.
Quantization Type: IQ4_XS
System: A clean Debian 12 install with only llama.cpp and ik_llama.cpp, single Xeon E5-2686v4 CPU, 128GB (16GBx8) DDR3 RAM, no GPU, no swap.
Command:
```bash
~/ik_llama.cpp/build_cpu/bin/llama-cli --no-mmap --model ~/unsloth/Qwen3-235B-A22B-128K-GGUF/IQ4_XS/Qwen3-235B-A22B-128K-IQ4_XS-00001-of-00003.gguf --threads 16 --ctx-size 16384 --seed 3407 --temp 0.6 --min-p 0.0 --top-p 0.95 --top-k 20 --prompt "<|im_start|>user\nCreate a Flappy Bird game in Python. You must include these things:\n1. You must use pygame.\n2. The background color should be randomly chosen and is a light shade. Start with a light blue color.\n3. Pressing SPACE multiple times will accelerate the bird.\n4. The bird's shape should be randomly chosen as a square, circle or triangle. The color should be randomly chosen as a dark color.\n5. Place on the bottom some land colored as dark brown or yellow chosen randomly.\n6. Make a score shown on the top right side. Increment if you pass pipes and don't hit them.\n7. Make randomly spaced pipes with enough space. Color them randomly as dark green or light brown or a dark gray shade.\n8. When you lose, show the best score. Make the text inside the screen. Pressing q or Esc will quit the game. Restarting is pressing SPACE again.\nThe final game should be inside a markdown section in Python. Check your code for errors and fix them before the final markdown section. /no_think<|im_end|>\n<|im_start|>assistant\n" -fa -rtr -fmoe
```

I am also experiencing an issue where the output briefly pauses for 1-2 seconds at commas. This problem does not occur with llama.cpp.

While it doesn't happen at every single comma, the output generation is perfectly smooth in parts of the text without commas.

Interestingly, if I add `9. Replace every comma with a pipe in python code` to the prompt, the pausing issue disappears.

What could be the problem?

1. with comma: https://youtu.be/n7tr2N_2DK8
2. without comma: https://youtu.be/Zy4r61EKq18

---

Additionally, I've confirmed this issue is present in the initial commit that added Qwen3 support: 9ba362706c998902752caf31d99fe077ed7d4faa.