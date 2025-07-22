### 🗣️ [#548](https://github.com/ikawrakow/ik_llama.cpp/discussions/548) - Poor performance with bf16 model on Qwen3 30B-A3B

| **Author** | `Gaolingx` |
| :--- | :--- |
| **Created** | 2025-06-22 |
| **Updated** | 2025-07-02 |

---

#### Description

## Introduction
I tried to run model [Qwen3-30B-A3B-GGUF](https://hf-mirror.com/unsloth/Qwen3-30B-A3B-GGUF) with ik_llama.cpp. Because I have a nvidia GPU(RTX 4060Ti) with 8G VRAM on my PC, so I compiled ik_llama.cpp with the cuda backend, and run with `-ot exps=CPU` to offload experts(ffn_down_exps, ffn_up_exps, gate_exps) to CPU.

Build options:
```text
cmake -B build -DGGML_NATIVE=OFF -DLLAMA_BUILD_SERVER=ON -DGGML_RPC=ON -DGGML_CUDA=ON -DGGML_AVX2=ON -DGGML_AVX512=OFF -DBUILD_SHARED_LIBS=ON
```

I tested `q8_0` quantization and `bf16` models, on `q8_0` model, the prompt processing speed(PP) the token generate speed(TG) are very quickly, I got a speed of up to 165 token/s PP and 18 token/s TG, that's a good start. but when I ran `bf16` model, the PP speed is much slower than before, It just 30-40token/s PP, 11-12 token/s TG, It's not even as good as only CPU ggml backend(about 51 token/s PP, 11 token/s TG), This performance is obviously not normal on bf16 models. It makes me confused. I've also found that the GPU spends quite a bit of time on the copy every time the token processing phase is processed, but quantization modes(like q8_0) don't have the above problem.

---
### cpu backend, `bf16` model(Qwen3-30B-A3B-BF16)
![ed1da34ea56ffe9a55fdc913fa17104f](https://github.com/user-attachments/assets/7df118ce-d21a-44ff-a4ee-e906dd9e9939)
---
### cuda backend, `bf16` model(Qwen3-30B-A3B-BF16)
![image](https://github.com/user-attachments/assets/34e3fc5c-ec54-45ea-a878-3af7d1a41793)
---
### cuda backend, `q8_0` model(Qwen3-30B-A3B-Q8_0)
![d1315e282e6c9ff022d8c85f8eb13c93](https://github.com/user-attachments/assets/08114f6f-8d8a-4030-9b51-617cd255dab2)

## System Info

Here are my SystemInfo(include hardware and software)

- Hardware
  - CPU: Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz(20c, 40t) x2
  - GPU: NVIDIA GeForce RTX 4060Ti 8G
  - RAM: RDIMM DDR4 2666 2Rx4 32G x16(12 Channels total)
  - Motherboard: Supermicro X11DPi-N
  - SSD: ZHITAI TiPlus7100 1TB
- Software
  - OS: Microsoft Windows 10 Pro
  - BIOS: Hyper-Threading-Enable, SNC-Disable
  - Model: Qwen3-235B-A22B-128K-Q8_0(unsloth/Qwen3-235B-A22B-128K-GGUF)
  - ik_llama.cpp:
  ```text
  ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
  ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
  ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4060 Ti, compute capability 8.9, VMM: yes
  INFO [                    main] build info | tid="54808" timestamp=1750526676 build=3761 commit="144ee1c4"
  INFO [                    main] system info | tid="54808" timestamp=1750526676 n_threads=16 n_threads_batch=-1 total_threads=40 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
  ```

## Benchmark

Here are the results of my initialllama-sweep-bench testing for PP speed and TG speed, the command line for is `ik_llama.cpp`

llama-sweep-bench:
```text
./llama-sweep-bench -m "%MODEL_PATH%" -c 16384 -t 16 -ngl 48 -fa -rtr -fmoe -ser 6,1 -ot exps=CPU
```

### ik_llama.cpp cuda backed (Model: Qwen3-30B-A3B-Q8_0)

<details>
main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 48, n_threads = 16, n_threads_batch = 16

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    2.308 |   221.84 |    6.463 |    19.81 |
|   512 |    128 |    512 |    2.437 |   210.11 |    7.741 |    16.54 |
|   512 |    128 |   1024 |    2.295 |   223.07 |    7.040 |    18.18 |
|   512 |    128 |   1536 |    2.537 |   201.81 |    7.739 |    16.54 |
|   512 |    128 |   2048 |    2.327 |   220.05 |    7.006 |    18.27 |
|   512 |    128 |   2560 |    2.523 |   202.97 |    7.766 |    16.48 |
|   512 |    128 |   3072 |    2.571 |   199.15 |    7.901 |    16.20 |
|   512 |    128 |   3584 |    2.531 |   202.26 |    7.717 |    16.59 |
|   512 |    128 |   4096 |    2.600 |   196.89 |    8.016 |    15.97 |
|   512 |    128 |   4608 |    2.602 |   196.80 |    7.962 |    16.08 |
|   512 |    128 |   5120 |    2.623 |   195.21 |    7.880 |    16.24 |
|   512 |    128 |   5632 |    2.614 |   195.86 |    8.090 |    15.82 |
|   512 |    128 |   6144 |    2.647 |   193.44 |    8.055 |    15.89 |
|   512 |    128 |   6656 |    2.647 |   193.43 |    7.963 |    16.07 |
|   512 |    128 |   7168 |    2.686 |   190.62 |    7.975 |    16.05 |
|   512 |    128 |   7680 |    2.687 |   190.54 |    8.069 |    15.86 |
|   512 |    128 |   8192 |    2.691 |   190.28 |    7.990 |    16.02 |
|   512 |    128 |   8704 |    2.713 |   188.69 |    8.030 |    15.94 |
|   512 |    128 |   9216 |    2.690 |   190.33 |    8.081 |    15.84 |
|   512 |    128 |   9728 |    2.706 |   189.24 |    8.015 |    15.97 |
|   512 |    128 |  10240 |    2.712 |   188.80 |    8.034 |    15.93 |
|   512 |    128 |  10752 |    2.777 |   184.35 |    8.097 |    15.81 |
|   512 |    128 |  11264 |    2.728 |   187.69 |    8.142 |    15.72 |
|   512 |    128 |  11776 |    2.651 |   193.15 |    8.040 |    15.92 |
|   512 |    128 |  12288 |    2.715 |   188.57 |    8.032 |    15.94 |
|   512 |    128 |  12800 |    2.727 |   187.76 |    8.091 |    15.82 |
|   512 |    128 |  13312 |    2.693 |   190.12 |    8.145 |    15.72 |
|   512 |    128 |  13824 |    2.692 |   190.22 |    8.137 |    15.73 |
|   512 |    128 |  14336 |    2.579 |   198.54 |    7.770 |    16.47 |
|   512 |    128 |  14848 |    2.688 |   190.45 |    8.211 |    15.59 |
|   512 |    128 |  15360 |    2.592 |   197.57 |    8.075 |    15.85 |
|   512 |    128 |  15872 |    2.660 |   192.47 |    8.132 |    15.74 |
</details>

### ik_llama.cpp cuda backed (Model: Qwen3-30B-A3B-BF16)

<details>
main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 48, n_threads = 16, n_threads_batch = 16

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   18.004 |    28.44 |   10.550 |    12.13 |
|   512 |    128 |    512 |   17.938 |    28.54 |   10.384 |    12.33 |
|   512 |    128 |   1024 |   17.859 |    28.67 |   10.370 |    12.34 |
|   512 |    128 |   1536 |   17.924 |    28.57 |   10.399 |    12.31 |
|   512 |    128 |   2048 |   17.989 |    28.46 |   10.386 |    12.32 |
|   512 |    128 |   2560 |   17.935 |    28.55 |   10.435 |    12.27 |
|   512 |    128 |   3072 |   18.006 |    28.44 |   10.513 |    12.18 |
|   512 |    128 |   3584 |   18.030 |    28.40 |   10.495 |    12.20 |
|   512 |    128 |   4096 |   18.063 |    28.35 |   10.578 |    12.10 |
|   512 |    128 |   4608 |   17.570 |    29.14 |   10.613 |    12.06 |
|   512 |    128 |   5120 |   17.685 |    28.95 |   10.600 |    12.08 |
|   512 |    128 |   5632 |   17.744 |    28.86 |   10.682 |    11.98 |
|   512 |    128 |   6144 |   17.911 |    28.59 |   10.640 |    12.03 |
|   512 |    128 |   6656 |   17.727 |    28.88 |   10.719 |    11.94 |
|   512 |    128 |   7168 |   17.529 |    29.21 |   10.636 |    12.03 |
|   512 |    128 |   7680 |   17.547 |    29.18 |   10.660 |    12.01 |
|   512 |    128 |   8192 |   17.517 |    29.23 |   10.708 |    11.95 |
|   512 |    128 |   8704 |   17.572 |    29.14 |   10.814 |    11.84 |
|   512 |    128 |   9216 |   17.542 |    29.19 |   10.813 |    11.84 |
|   512 |    128 |   9728 |   17.615 |    29.07 |   10.815 |    11.84 |
|   512 |    128 |  10240 |   17.573 |    29.14 |   10.839 |    11.81 |
|   512 |    128 |  10752 |   17.616 |    29.06 |   10.858 |    11.79 |
|   512 |    128 |  11264 |   17.670 |    28.98 |   10.899 |    11.74 |
|   512 |    128 |  11776 |   17.764 |    28.82 |   11.194 |    11.44 |
|   512 |    128 |  12288 |   17.622 |    29.05 |   10.960 |    11.68 |
|   512 |    128 |  12800 |   17.658 |    28.99 |   11.039 |    11.60 |
|   512 |    128 |  13312 |   17.661 |    28.99 |   11.036 |    11.60 |
|   512 |    128 |  13824 |   17.624 |    29.05 |   11.093 |    11.54 |
|   512 |    128 |  14336 |   17.587 |    29.11 |   11.094 |    11.54 |
|   512 |    128 |  14848 |   17.650 |    29.01 |   11.174 |    11.45 |
|   512 |    128 |  15360 |   17.648 |    29.01 |   11.190 |    11.44 |
|   512 |    128 |  15872 |   17.645 |    29.02 |   11.204 |    11.42 |

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2025-06-22** at **15:16:00**:<br>

Don't use `-rtr` for the `bf16` model.

> 👤 **Gaolingx** replied the **2025-06-22** at **15:31:07**:<br>
> wow, thanks a lot for your suggestion, the speed is normal now, I got ~65 PP speed and ~11.8 TG speed, but the cpu+cuda(`-ot exps=CPU`) speed doesn't seem to be much faster than the pure cpu, although it is a moe model. maybe I should do a more detailed benchmark.
> 
> 👤 **ikawrakow** replied the **2025-06-22** at **15:35:13**:<br>
> You need larger u-batch size for better PP performance. The experts are in RAM and need to be offloaded to the GPU, which takes a while. If you run `llama-sweep-bench` with `-ub 2048` you will see much better PP performance.
> 
> 👤 **Gaolingx** replied the **2025-07-02** at **10:54:41**:<br>
> Hi, we all know that runtime repack(`-rtr`) is good to use with hybrid GPU + CPU, according to my research in the last few days, if we don't add the '-rtr' parameter, when we input long prompts, the cuda device needs to spend a long time on copying (and you can see in the task manager that the usage of 'Copy1' is quite high, but the usage of `CPU` and `CUDA` is insufficient), and the processing speed of prompt words is also significantly lower than the performance with the '-rtr' parameter, or even worse than the cpu only, what is the reason for this?
> ![09f30b1c-8174-43f0-8b7e-113ec8bbe4dd](https://github.com/user-attachments/assets/ac09c33d-f102-4e89-8c9f-b541d562a902)
> 
> 👤 **ikawrakow** replied the **2025-07-02** at **12:14:25**:<br>
> I'm not sure I understand what could be the issue from the description. Can you tell us what is the model you are using and post your command line?
> 
> 👤 **Gaolingx** replied the **2025-07-02** at **14:23:58**:<br>
> > I'm not sure I understand what could be the issue from the description. Can you tell us what is the model you are using and post your command line?
> 
> Ok. I ran llama-sweep-bench again and tested the 16k context length data of three sets of qwen3 30ba3b models. They are that the q8_0 model with `-rtr` parameter, the q8_0 model without `-rtr` parameter, and the bf16 model without `-rtr` parameter. To control the variables, in the test group without the -rtr parameter, I added the `--no-mmap` parameter. The rest of the startup parameters remained the same. The  llama-sweep-bench startup parameters and test results are as follows.
> 
> I have come to the conclusion that, whether it is the q8_0 or bf16 model, on my platform, if the `-rtr` parameter is not used, the prompt processing performance during cpu+gpu hybrid inference will be significantly affected. The larger the model, the more serious this situation is. However, The token generation speed is normal and in line with expectations. What causes this? How does the runtime repack tensors(-rtr) to improve prompt processing performance?
> 
> ---
> ## ik_llama.cpp cuda backed (Model: Qwen3-30B-A3B-Q8_0 with `-rtr`)
> 
> <details>
> <summary>./llama-sweep-bench -m "D:\Downloads\Qwen3-30B-A3B-Q8_0.gguf" -c 16384 -t 16 -ngl 49 -fa -rtr -fmoe -ser 6,1 -ot exps=CPU</summary>
> 
> main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 49, n_threads = 16, n_threads_batch = 16
> 
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |   512 |    128 |      0 |    2.247 |   227.87 |    5.941 |    21.54 |
> |   512 |    128 |    512 |    2.293 |   223.28 |    6.718 |    19.05 |
> |   512 |    128 |   1024 |    2.354 |   217.46 |    6.981 |    18.34 |
> |   512 |    128 |   1536 |    2.382 |   214.94 |    7.088 |    18.06 |
> |   512 |    128 |   2048 |    2.406 |   212.81 |    7.011 |    18.26 |
> |   512 |    128 |   2560 |    2.394 |   213.84 |    7.078 |    18.09 |
> |   512 |    128 |   3072 |    2.408 |   212.61 |    7.080 |    18.08 |
> |   512 |    128 |   3584 |    2.383 |   214.83 |    7.127 |    17.96 |
> |   512 |    128 |   4096 |    2.415 |   211.97 |    7.083 |    18.07 |
> |   512 |    128 |   4608 |    2.391 |   214.12 |    7.170 |    17.85 |
> |   512 |    128 |   5120 |    2.461 |   208.03 |    7.216 |    17.74 |
> |   512 |    128 |   5632 |    2.448 |   209.11 |    7.233 |    17.70 |
> |   512 |    128 |   6144 |    2.458 |   208.31 |    7.286 |    17.57 |
> |   512 |    128 |   6656 |    2.456 |   208.48 |    7.251 |    17.65 |
> |   512 |    128 |   7168 |    2.413 |   212.17 |    7.160 |    17.88 |
> |   512 |    128 |   7680 |    2.450 |   208.98 |    7.310 |    17.51 |
> |   512 |    128 |   8192 |    2.482 |   206.26 |    7.302 |    17.53 |
> |   512 |    128 |   8704 |    2.365 |   216.50 |    7.130 |    17.95 |
> |   512 |    128 |   9216 |    2.371 |   215.94 |    7.109 |    18.01 |
> |   512 |    128 |   9728 |    2.381 |   214.99 |    7.264 |    17.62 |
> |   512 |    128 |  10240 |    2.395 |   213.81 |    7.192 |    17.80 |
> |   512 |    128 |  10752 |    2.402 |   213.16 |    7.103 |    18.02 |
> |   512 |    128 |  11264 |    2.402 |   213.18 |    7.005 |    18.27 |
> |   512 |    128 |  11776 |    2.372 |   215.87 |    7.023 |    18.22 |
> |   512 |    128 |  12288 |    2.474 |   206.92 |    6.762 |    18.93 |
> |   512 |    128 |  12800 |    2.457 |   208.42 |    6.808 |    18.80 |
> |   512 |    128 |  13312 |    2.442 |   209.64 |    6.740 |    18.99 |
> |   512 |    128 |  13824 |    2.447 |   209.22 |    6.824 |    18.76 |
> |   512 |    128 |  14336 |    2.473 |   207.03 |    6.704 |    19.09 |
> |   512 |    128 |  14848 |    2.524 |   202.86 |    6.695 |    19.12 |
> |   512 |    128 |  15360 |    2.573 |   199.00 |    7.093 |    18.05 |
> |   512 |    128 |  15872 |    2.520 |   203.17 |    6.611 |    19.36 |
> 
> </details>
> 
> ---
> ## ik_llama.cpp cuda backed (Model: Qwen3-30B-A3B-Q8_0 without `-rtr`)
> 
> <details>
> <summary>./llama-sweep-bench -m "D:\Downloads\Qwen3-30B-A3B-Q8_0.gguf" -c 16384 -t 16 -ngl 49 -fa --no-mmap -fmoe -ser 6,1 -ot exps=CPU</summary>
> 
> main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 49, n_threads = 16, n_threads_batch = 16
> 
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |   512 |    128 |      0 |    9.527 |    53.74 |    6.171 |    20.74 |
> |   512 |    128 |    512 |    9.556 |    53.58 |    6.117 |    20.93 |
> |   512 |    128 |   1024 |    9.554 |    53.59 |    6.184 |    20.70 |
> |   512 |    128 |   1536 |    9.551 |    53.61 |    6.149 |    20.82 |
> |   512 |    128 |   2048 |    9.590 |    53.39 |    6.255 |    20.46 |
> |   512 |    128 |   2560 |    9.523 |    53.76 |    6.230 |    20.55 |
> |   512 |    128 |   3072 |    9.509 |    53.84 |    6.257 |    20.46 |
> |   512 |    128 |   3584 |    9.555 |    53.58 |    6.274 |    20.40 |
> |   512 |    128 |   4096 |    9.640 |    53.11 |    6.705 |    19.09 |
> |   512 |    128 |   4608 |    9.638 |    53.12 |    6.409 |    19.97 |
> |   512 |    128 |   5120 |    9.615 |    53.25 |    6.388 |    20.04 |
> |   512 |    128 |   5632 |    9.652 |    53.04 |    6.360 |    20.12 |
> |   512 |    128 |   6144 |    9.662 |    52.99 |    6.430 |    19.91 |
> |   512 |    128 |   6656 |    9.702 |    52.77 |    6.480 |    19.75 |
> |   512 |    128 |   7168 |    9.609 |    53.28 |    6.494 |    19.71 |
> |   512 |    128 |   7680 |    9.606 |    53.30 |    6.485 |    19.74 |
> |   512 |    128 |   8192 |    9.622 |    53.21 |    6.521 |    19.63 |
> |   512 |    128 |   8704 |    9.620 |    53.22 |    6.546 |    19.55 |
> |   512 |    128 |   9216 |    9.559 |    53.56 |    6.602 |    19.39 |
> |   512 |    128 |   9728 |    9.538 |    53.68 |    6.542 |    19.57 |
> |   512 |    128 |  10240 |    9.563 |    53.54 |    6.626 |    19.32 |
> |   512 |    128 |  10752 |    9.610 |    53.28 |    6.561 |    19.51 |
> |   512 |    128 |  11264 |    9.689 |    52.85 |    6.618 |    19.34 |
> |   512 |    128 |  11776 |    9.619 |    53.23 |    6.628 |    19.31 |
> |   512 |    128 |  12288 |    9.654 |    53.03 |    6.452 |    19.84 |
> |   512 |    128 |  12800 |    9.800 |    52.24 |    6.578 |    19.46 |
> |   512 |    128 |  13312 |    9.641 |    53.11 |    6.613 |    19.35 |
> |   512 |    128 |  13824 |    9.638 |    53.12 |    6.513 |    19.65 |
> |   512 |    128 |  14336 |    9.686 |    52.86 |    6.555 |    19.53 |
> |   512 |    128 |  14848 |    9.729 |    52.62 |    6.609 |    19.37 |
> |   512 |    128 |  15360 |    9.702 |    52.77 |    6.624 |    19.32 |
> |   512 |    128 |  15872 |    9.697 |    52.80 |    6.636 |    19.29 |
> 
> </details>
> 
> ---
> ## ik_llama.cpp cuda backed (Model: Qwen3-30B-A3B-BF16 without `-rtr`)
> 
> <details>
> <summary>./llama-sweep-bench -m "D:\Downloads\unsloth\Qwen3-30B-A3B-GGUF\BF16\Qwen3-30B-A3B-BF16-00001-of-00002.gguf" -c 16384 -t 16 -ngl 49 -fa --no-mmap -fmoe -ser 6,1 -ot exps=CPU</summary>
> 
> main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 49, n_threads = 16, n_threads_batch = 16
> 
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |   512 |    128 |      0 |   17.771 |    28.81 |    9.791 |    13.07 |
> |   512 |    128 |    512 |   17.398 |    29.43 |    9.025 |    14.18 |
> |   512 |    128 |   1024 |   17.305 |    29.59 |    9.030 |    14.17 |
> |   512 |    128 |   1536 |   17.367 |    29.48 |    9.054 |    14.14 |
> |   512 |    128 |   2048 |   17.859 |    28.67 |    9.342 |    13.70 |
> |   512 |    128 |   2560 |   17.700 |    28.93 |    9.143 |    14.00 |
> |   512 |    128 |   3072 |   17.696 |    28.93 |    9.170 |    13.96 |
> |   512 |    128 |   3584 |   17.939 |    28.54 |    9.241 |    13.85 |
> |   512 |    128 |   4096 |   17.926 |    28.56 |    9.212 |    13.89 |
> |   512 |    128 |   4608 |   17.714 |    28.90 |    9.280 |    13.79 |
> |   512 |    128 |   5120 |   17.822 |    28.73 |    9.226 |    13.87 |
> |   512 |    128 |   5632 |   17.830 |    28.72 |    9.273 |    13.80 |
> |   512 |    128 |   6144 |   17.749 |    28.85 |    9.121 |    14.03 |
> |   512 |    128 |   6656 |   17.581 |    29.12 |    9.356 |    13.68 |
> |   512 |    128 |   7168 |   17.517 |    29.23 |    9.401 |    13.62 |
> |   512 |    128 |   7680 |   17.408 |    29.41 |    9.393 |    13.63 |
> |   512 |    128 |   8192 |   17.451 |    29.34 |    9.371 |    13.66 |
> |   512 |    128 |   8704 |   17.409 |    29.41 |    9.544 |    13.41 |
> |   512 |    128 |   9216 |   17.443 |    29.35 |    9.476 |    13.51 |
> |   512 |    128 |   9728 |   17.449 |    29.34 |   10.037 |    12.75 |
> |   512 |    128 |  10240 |   17.370 |    29.48 |    9.480 |    13.50 |
> |   512 |    128 |  10752 |   17.472 |    29.30 |    9.504 |    13.47 |
> |   512 |    128 |  11264 |   17.612 |    29.07 |    9.500 |    13.47 |
> |   512 |    128 |  11776 |   17.492 |    29.27 |    9.580 |    13.36 |
> |   512 |    128 |  12288 |   17.384 |    29.45 |    9.569 |    13.38 |
> |   512 |    128 |  12800 |   18.000 |    28.44 |    9.436 |    13.56 |
> |   512 |    128 |  13312 |   17.759 |    28.83 |    9.493 |    13.48 |
> |   512 |    128 |  13824 |   17.905 |    28.60 |    9.442 |    13.56 |
> |   512 |    128 |  14336 |   17.843 |    28.69 |    9.372 |    13.66 |
> |   512 |    128 |  14848 |   17.928 |    28.56 |    9.538 |    13.42 |
> |   512 |    128 |  15360 |   17.902 |    28.60 |    9.436 |    13.57 |
> |   512 |    128 |  15872 |   17.971 |    28.49 |    9.336 |    13.71 |
> 
> </details>
> 
> 👤 **ikawrakow** replied the **2025-07-02** at **14:40:43**:<br>
> When you use `-rtr`, the tensors not offloaded to the GPU get repacked to a row-interleaved version. `Q8_0` becomes `Q8_0_R8`, and `BF16` becomes `BF16_R16`. `Q8_0_R8` and `BF16_R16` are not supported by the CUDA backend, so matrix multiplications with these tensors are done on the CPU. When you do not use `-rtr`, there is no repacking, CUDA supports `Q8_0` and `BF16`, so the tensors stored in RAM get copied to the GPU to do matrix multiplications. If the model is large, and your PCI-E is not very fast, the copying to VRAM takes a long time, so your PP performance becomes low. You can improve the performance by using larger u-batches because more work is done per copy to the GPU (tensors are copied once, but multiply 2048 tokens with `-ub 2048`. To accomplish the same with the u-batch of 512 you are using, tensors need to get copied 4 times). If you don't want to repack, and don't want to use larger u-batches, you can prevent copying to the GPU using `-op 26,0,27,0,29,0`. In that case `bf16` performance will be slightly lower than with `-rtr`, but `Q8_0` performance will be somewhere in the middle between `-rtr` and no `-rtr`.