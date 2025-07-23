### 🐛 [#490](https://github.com/ikawrakow/ik_llama.cpp/issues/490) - Bug: Performance drop with 14292913 [#461](https://github.com/ikawrakow/ik_llama.cpp/issues/461)

| **Author** | `nux` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-03 |
| **Updated** | 2025-06-05 |

---

#### Description

### What happened?

Performance dropping with commit 14292913 #461

To identify which commit the performance dropped with I was running:

Was running for i in `cut -d " " -f1 commits.txt `;do git checkout $i;./cmd-build.sh ;./start-bench.sh >> results.txt;done

start-bench.sh is: 
./build/bin/llama-bench -m /mnt/nvme/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4/DeepSeek-V3-0324-IQ4_K_R4-00001-of-00010.gguf -p 512 -t 32 -mla 2 -fa 1 -fmoe 1 -ngl 99 --override-tensor "exps=CPU" -amb 512

Relevant results.txt:

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

build: c7ecd4e2 (3712)

 
Building like this:
cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build build --config Release -j --clean-first

Running on 2x9115, 768gb ram, 3090 gpu



### Name and Version

version: 3710 (9fb82af3)
built with cc (Debian 12.2.0-14+deb12u1) 12.2.0 for x86_64-linux-gnu


### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-06-03** at **14:24:50**:<br>

Are all tensors `IQ4_K_R4`? If not, what is the quantization mix in this model?

---

👤 **nux** commented the **2025-06-03** at **14:30:39**:<br>

This is https://huggingface.co/ubergarm/DeepSeek-V3-0324-GGUF IQ4_K_R4

They are not all IQ4_K_R4 - I believe this is summary:

llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  612 tensors
llama_model_loader: - type iq4_k_r4:  116 tensors
llama_model_loader: - type iq5_k_r4:   58 tensors

---

👤 **ikawrakow** commented the **2025-06-03** at **15:08:10**:<br>

I cannot run DeepSeek-V3, but as a surrogate here some results with Qwen3-30B-A22B. Quantized with the same mix of `IQ4_K_R4` and `IQ5_K_R4` for the experts, `Q8_0` everything else, just like the model you have. My system is Ryzen-7950X + RTX-4080. I'm leaving all experts on the CPU (`-ot exps=CPU`).

To make things more interesting I'm using `pp2048` instead of `pp512`.

The "good" build 24c010b3

| model                          |       size |     params | backend    | ngl | n_ubatch | fa |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -: | ------------: | ---------------: |
| qwen3moe ?B IQ4_K_R4 - 4.5 bpw |  17.57 GiB |    30.53 B | CUDA       |  99 |      512 |  1 |        pp2048 |    606.31 ± 3.88 |
| qwen3moe ?B IQ4_K_R4 - 4.5 bpw |  17.57 GiB |    30.53 B | CUDA       |  99 |     1024 |  1 |        pp2048 |    622.61 ± 8.59 |
| qwen3moe ?B IQ4_K_R4 - 4.5 bpw |  17.57 GiB |    30.53 B | CUDA       |  99 |     2048 |  1 |        pp2048 |   616.80 ± 7.54 |
| qwen3moe ?B IQ4_K_R4 - 4.5 bpw |  17.57 GiB |    30.53 B | CUDA       |  99 |     1024 |  1 |         tg128 |     34.48 ± 0.03 |

And now the "bad" build (f6d5fbdc, which is latest master)

| model                          |       size |     params | backend    | ngl | n_ubatch | fa |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -: | ------------: | ---------------: |
| qwen3moe ?B IQ4_K_R4 - 4.5 bpw |  17.57 GiB |    30.53 B | CUDA       |  99 |      512 |  1 |        pp2048 |    481.03 ± 3.55 |
| qwen3moe ?B IQ4_K_R4 - 4.5 bpw |  17.57 GiB |    30.53 B | CUDA       |  99 |     1024 |  1 |        pp2048 |    893.92 ± 1.59 |
| qwen3moe ?B IQ4_K_R4 - 4.5 bpw |  17.57 GiB |    30.53 B | CUDA       |  99 |     2048 |  1 |        pp2048 |   1554.57 ± 2.93 |
| qwen3moe ?B IQ4_K_R4 - 4.5 bpw |  17.57 GiB |    30.53 B | CUDA       |  99 |      512 |  1 |         tg128 |     34.45 ± 0.41 |
| qwen3moe ?B IQ4_K_R4 - 4.5 bpw |  17.57 GiB |    30.53 B | CUDA       |  99 |     1024 |  1 |         tg128 |     34.50 ± 0.27 |
| qwen3moe ?B IQ4_K_R4 - 4.5 bpw |  17.57 GiB |    30.53 B | CUDA       |  99 |     2048 |  1 |         tg128 |     34.69 ± 0.01 |

I see zero difference in TG. PP on main is indeed slower for u-batch of 512, but becomes 2.5X faster for u-batch = 2048!

---

👤 **ikawrakow** commented the **2025-06-03** at **15:36:46**:<br>

If you say that you don't want to use large u-batches because of something, you can recover the pre-#461 behavior using `-op 26,0,27,0,29,0`. This disables offloading of tensors that are on the CPU to the GPU. This has not been implemented in `llama-bench`, which has its own command line argument parsing, but is available in `llama-sweep-bench`.

Here is what I get with
```
./bin/llama-sweep-bench -m $model -c 16384 -up 2048 -t 16 -ngl 100 -ot exps=CPU
```

### "Good build"

 |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |    3.158 |   648.45 |   14.698 |    34.84 |
|  2048 |    512 |   2048 |    3.275 |   625.28 |   14.792 |    34.61 |
|  2048 |    512 |   4096 |    3.235 |   633.05 |   15.047 |    34.03 |
|  2048 |    512 |   6144 |    3.262 |   627.77 |   15.252 |    33.57 |
|  2048 |    512 |   8192 |    3.308 |   619.06 |   15.425 |    33.19 |
|  2048 |    512 |  10240 |    3.368 |   608.10 |   15.702 |    32.61 |
|  2048 |    512 |  12288 |    4.105 |   498.92 |   15.776 |    32.45 |
|  2048 |    512 |  14336 |    3.596 |   569.58 |   15.549 |    32.93 |
 
### Main branch

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |    1.352 |  1514.60 |   14.926 |    34.30 |
|  2048 |    512 |   2048 |    1.345 |  1523.06 |   15.034 |    34.06 |
|  2048 |    512 |   4096 |    1.378 |  1486.27 |   15.232 |    33.61 |
|  2048 |    512 |   6144 |    1.413 |  1449.21 |   15.413 |    33.22 |
|  2048 |    512 |   8192 |    1.445 |  1417.62 |   15.612 |    32.79 |
|  2048 |    512 |  10240 |    1.482 |  1381.74 |   15.875 |    32.25 |
|  2048 |    512 |  12288 |    1.516 |  1350.95 |   15.973 |    32.05 |
|  2048 |    512 |  14336 |    1.546 |  1324.99 |   16.158 |    31.69 |

### Main branch with -op 26,0,27,0,29,0

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |    3.293 |   621.93 |   14.868 |    34.44 |
|  2048 |    512 |   2048 |    3.588 |   570.87 |   15.029 |    34.07 |
|  2048 |    512 |   4096 |    3.452 |   593.34 |   15.157 |    33.78 |
|  2048 |    512 |   6144 |    3.463 |   591.43 |   15.380 |    33.29 |
|  2048 |    512 |   8192 |    3.359 |   609.71 |   15.564 |    32.90 |
|  2048 |    512 |  10240 |    3.375 |   606.87 |   15.802 |    32.40 |
|  2048 |    512 |  12288 |    3.622 |   565.51 |   15.918 |    32.17 |
|  2048 |    512 |  14336 |    3.439 |   595.48 |   15.675 |    32.66 |

---

👤 **nux** commented the **2025-06-03** at **22:24:24**:<br>

I don't mind using larger batch sizes. I mostly leave things as they are when it's working and only look at it when there's a problem :-D

That is good to know with ubatch. It seems to work very well for qwen3

nux@red ~/dev/ik_llama.cpp $ ./build/bin/llama-bench -m /mnt/nvme/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf -p 2048 -t 32 -mla 3 -fa 1 -fmoe 1 -ngl 99 -amb 512 -ub 512,1024,2048 -ot blk\.1[2-9]\.ffn.*=CPU -ot blk\.[2-8][0-9]\.ffn.*=CPU -ot blk\.9[0-3]\.ffn.*=CPU
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
| model                          |       size |     params | backend    | ngl | n_ubatch | fa | mla |   amb | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -: | --: | ----: | ---: | ------------: | ---------------: |
| qwen3moe ?B IQ3_K - 3.4325 bpw | 106.83 GiB |   235.09 B | CUDA       |  99 |      512 |  1 |   3 |   512 |    1 |        pp2048 |   103.22 ± 14.97 |
| qwen3moe ?B IQ3_K - 3.4325 bpw | 106.83 GiB |   235.09 B | CUDA       |  99 |      512 |  1 |   3 |   512 |    1 |         tg128 |     19.01 ± 0.01 |
| qwen3moe ?B IQ3_K - 3.4325 bpw | 106.83 GiB |   235.09 B | CUDA       |  99 |     1024 |  1 |   3 |   512 |    1 |        pp2048 |    195.53 ± 0.19 |
| qwen3moe ?B IQ3_K - 3.4325 bpw | 106.83 GiB |   235.09 B | CUDA       |  99 |     1024 |  1 |   3 |   512 |    1 |         tg128 |     18.92 ± 0.05 |
| qwen3moe ?B IQ3_K - 3.4325 bpw | 106.83 GiB |   235.09 B | CUDA       |  99 |     2048 |  1 |   3 |   512 |    1 |        pp2048 |    321.14 ± 0.48 |
| qwen3moe ?B IQ3_K - 3.4325 bpw | 106.83 GiB |   235.09 B | CUDA       |  99 |     2048 |  1 |   3 |   512 |    1 |         tg128 |     18.49 ± 0.55 |

build: f6d5fbdc (3725)


If I'm the only one having problems, I'll keep using 24c010b3 for deepseek-r1 and deepseek-v3.

---

👤 **ikawrakow** commented the **2025-06-04** at **04:47:10**:<br>

>If I'm the only one having problems, I'll keep using https://github.com/ikawrakow/ik_llama.cpp/commit/24c010b3916b5f1bb9d712d610d1fe9308ef7df4 for deepseek-r1 and deepseek-v3.

Did you try any of the options available to you with DeepSeek?

I'll close the issue then.

---

👤 **nux** commented the **2025-06-04** at **05:47:54**:<br>

What do you mean options available with DeepSeek? I tried ubatch and have been running mla 3. 

Would any of them cause this decrease in performance for this command? ~10t/s to ~4.8t/s
./build/bin/llama-bench -m /mnt/nvme/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_K_R4/DeepSeek-V3-0324-IQ4_K_R4-00001-of-00010.gguf -p 512 -t 32 -mla 2 -fa 1 -fmoe 1 -ngl 99 --override-tensor "exps=CPU" -amb 512

This issue came up originally when trying to figure out why ubergarm's deepseek-r1 was performing poorly. The older deepseek-v3 benchmarks that i had sitting around in a .txt made it easy to compare.

If you would like me to try anything specific I can, but I don't know where to start diagnosing my issue any further

I wouldn't consider the issue resolved. Using commit 24c010b3 for deepseek seems more of a short term workaround than resolution.   

That being said I don't think we pay you enough. I appreciate all the work you've done.

---

👤 **ikawrakow** commented the **2025-06-04** at **05:52:12**:<br>

I didn't see your performance values for `-ub 2048` (or even `-b 4096 -ub 4096`

Neither did I see results for your regular way of using DeepSeek but adding `-op 26,0,27,0,29,0` to your command line. This latter option should match what you had prior to #461.

---

👤 **nux** commented the **2025-06-05** at **13:53:10**:<br>

-op 26,0,27,0,29,0 brought back the performance. I hadn't tried that one as my PCI-E speed is 16x - but working now.

Thanks