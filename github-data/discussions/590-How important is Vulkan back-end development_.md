### 🗣️ [#590](https://github.com/ikawrakow/ik_llama.cpp/discussions/590) - How important is Vulkan back-end development?

| **Author** | `ikawrakow` |
| :--- | :--- |
| **Created** | 2025-07-06 |
| **Updated** | 2025-07-18 |

---

#### Description

Tthe Vulkan back-end in `ik_llama.cpp` is now usable, and performance is better than `llama.cpp` (see, e.g., PR #584 that has a comparison for a MoE model). But compared to CUDA on the same GPU, performance is much lower, especially for MoE models (and most users appear to be using `ik_llama.cpp` exactly for one of the giant MoE models). I have mixed feelings how to proceed:
* There is much more performance optimization potential in the Vulkan back-end compared to CUDA or CPU. So, from that point of view it seems worthwhile to put some effort into optimizing the Vulkan back-end
* I know nothing about Vulkan programming in general or the `llama.cpp` Vulkan back-end in particular, hence, at least initially, it will be an uphill battle. Without a significant interest from the user base, I don't feel particularly motivated to do this to myself.

---

#### 🗣️ Discussion

👤 **OneOfOne** replied the **2025-07-06** at **16:55:32**:<br>

On AMD, vulkan is faster and more memory efficient than rocm.

---

👤 **mcm007** replied the **2025-07-06** at **18:25:18**:<br>

Currently, owners of Nvidia GPUs have access to a wide range of inference engines (e.g., vllm, exllama, sglang, mlc, aphrodite-engine) that are optimized for CUDA. This allows them to fully utilize their hardware, which is great.

In contrast, Vulkan support could provide significant benefits to users of AMD and Intel GPUs, which currently have less mature tooling and support.

AMD appears not so friendly toward regular consumers, eg. AMD Rocm barely supports their top GPUs.
The recent Vulkan improvements by jeffbolznv on mainline llama.cpp are higher for Nvidia GPUs because he seems from Nvidia backgrounds.
Is not nice that we don't notice AMD people providing some support... just enough to be noticed.
As much I don't like Nvidia I swapped my new 7900XTX for a used 3090.

Also, with Vulkan support would be possible to run the fast `ik_llama.cpp` on devices like Intel iGPU or Ryzen 3400G APU, using `KS` quants, re-use the quantized files, etc.

I want to acknowledge the effort and quality of your work, therefore whatever you choose (improve speed, quants quality, Vulkan, features, ...) doesn't matter at the end: they will benefit us, users/community.

> 👤 **saood06** replied the **2025-07-06** at **23:34:55**:<br>
> >Currently, owners of Nvidia GPUs have access to a wide range of inference engines (e.g., vllm, exllama, sglang, mlc, aphrodite-engine) that are optimized for CUDA. This allows them to fully utilize their hardware, which is great.
> 
> All of the ones you list do offer some form of AMD support: [vllm](https://docs.vllm.ai/en/v0.6.3/getting_started/amd-installation.html), exllama V2 with [builds](https://github.com/turboderp-org/exllamav2/releases/tag/v0.3.1) for rocm and plans for it in v3, [sglang](https://docs.sglang.ai/references/amd.html), [mlc](https://github.com/mlc-ai/mlc-llm) table shows both Vulkan and ROCm support, [aphrodite-engine](https://aphrodite.pygmalion.chat/installation/installation-rocm/).
> 
> > As much I don't like Nvidia I swapped my new 7900XTX for a used 3090.
> 
> To be transparent, I don't own a modern AMD card, and I do own a 3090, so I have no personal experience using ROCm. But at least it looks like there is support for AMD to some degree in everything you listed.
> 
> >Ryzen 3400G APU, using KS quants, re-use the quantized files, etc.
> 
> But I have owned and used 3400G (upgraded past it). I'm not sure if the iGPU would be better (or at least better enough to matter) than the AVX2 CPU backend, what I miss about the iGPU, is that it lets you run without discrete GPU (or fully passing it through to a VM).
> 
> 👤 **mcm007** replied the **2025-07-07** at **05:45:28**:<br>
> > All of the ones you list do offer some form of AMD support: [vllm](https://docs.vllm.ai/en/v0.6.3/getting_started/amd-installation.html), exllama V2 with [builds](https://github.com/turboderp-org/exllamav2/releases/tag/v0.3.1) for rocm and plans for it in v3, [sglang](https://docs.sglang.ai/references/amd.html), [mlc](https://github.com/mlc-ai/mlc-llm) table shows both Vulkan and ROCm support, [aphrodite-engine](https://aphrodite.pygmalion.chat/installation/installation-rocm/).
> 
> Usually, support is not complete and misses features or optimizations like FA, supporting all quants, and quantized cache. :disappointed:
> 
> > But I have owned and used 3400G (upgraded past it). I'm not sure if the iGPU would be better (or at least better enough to matter) than the AVX2 CPU backend, what I miss about the iGPU, is that it lets you run without discrete GPU (or fully passing it through to a VM).
> 
> Since IK created `-rtr`, or with the recent on-the-fly repacks #531, #533, #534, PP performance has skyrocketed, making the CPU viable for small models on simple tasks :smile:.
> Indeed, the extra performance added by iGPU part doesn't seem worth the effort, but for models small enough to fit in the default 2GB* allocated memory, the sweep-bench looks incredible on the Vulkan build:
> ![performance_comparison_pp](https://github.com/user-attachments/assets/4a10476e-b9cf-47a3-abbf-06a6bf92444d)
> ![performance_comparison_tg](https://github.com/user-attachments/assets/8a9746e6-7dcd-4dcf-a19e-54a1b14b2f10)
> 
> * There is a way to increase the memory allocated to the iGPU [Smokeless_UMAF](https://github.com/DavidS95/Smokeless_UMAF) but it's a bit of a hassle - one needs to boot from the modified BIOS every time and make the modification.
> 
> 👤 **saood06** replied the **2025-07-07** at **06:09:54**:<br>
> > Usually, support is not complete and misses features or optimizations like FA, supporting all quants, and quantized cache. 😞
> 
> I did look into the state of flash attention support for ROCm and it did seem like they are working on it with things like paged attention not fully there yet.
> 
> Like I said I don't have personal experience so I don't know what the experience is like, just thought it should be mentioned that they all do seem like they do support the hardware (to some level).
> 
> > Since IK created `-rtr`, or with the recent on-the-fly repacks #531, #533, #534, PP performance has skyrocketed, making the CPU viable for small models on simple tasks 😄. Indeed, the extra performance added by iGPU part doesn't seem worth the effort.
> 
> Yeah.
> 
> >the sweep-bench looks incredible on the Vulkan build
> 
> Thanks for the graphs. I'd be curious about peak batched performance comparisons (I never got around to adding a plot tool to `batched-bench`)
> 
> >There is a way to increase the memory allocated to the iGPU [Smokeless_UMAF](https://github.com/DavidS95/Smokeless_UMAF) but it's a bit of a hassle - one needs to boot from the modified BIOS every time and make the modification.
> 
> That is interesting to hear for if I ever use that CPU again (but if I do use it, I'm not sure if I'd want to allocate more or less VRAM assuming less is possible).
> 
> 👤 **ikawrakow** replied the **2025-07-07** at **07:16:56**:<br>
> @mcm007  What is the CPU for these graphs? PP < 200 t/s seems quite low for a 0.6B model.
> 
> Here is what I get for `Q6_K`-quantized Qwen3-0.6B on my Ryzen-7950X CPU:
> 
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |   512 |    128 |      0 |    0.201 |  2546.00 |    1.259 |   101.65 |
> |   512 |    128 |    512 |    0.209 |  2451.59 |    1.463 |    87.48 |
> |   512 |    128 |   1024 |    0.233 |  2197.58 |    1.646 |    77.78 |
> |   512 |    128 |   1536 |    0.258 |  1985.52 |    1.669 |    76.67 |
> |   512 |    128 |   2048 |    0.282 |  1814.45 |    1.715 |    74.63 |
> |   512 |    128 |   2560 |    0.307 |  1665.39 |    1.783 |    71.80 |
> |   512 |    128 |   3072 |    0.333 |  1537.27 |    1.856 |    68.95 |
> |   512 |    128 |   3584 |    0.361 |  1419.98 |    1.925 |    66.48 |
> 
> 👤 **mcm007** replied the **2025-07-07** at **09:01:59**:<br>
> @saood06
> 
> > I'd be curious about peak batched performance comparisons (I never got around to adding a plot tool to batched-bench)
> 
> 
> 
> <details>
> <summary>Results here, click to expand</summary>
> 
> ### Vulkan build
> 
> `llama-batched-bench -m /models1/Qwen_Qwen3-0.6B-Q6_K.gguf -c 4096 -b 512 -ub 512 -ngl 0,1 -npp 128 -ntg 128 -npl 1,2,4,6,8,10,12,14,16`
> 
> |    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
> |-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
> |   128 |    128 |    1 |    256 |    2.158 |    59.33 |    3.076 |    41.61 |    5.233 |    48.92 |
> |   128 |    128 |    2 |    512 |    1.814 |   141.12 |    4.738 |    54.03 |    6.552 |    78.14 |
> |   128 |    128 |    4 |   1024 |    1.870 |   273.78 |    7.437 |    68.84 |    9.308 |   110.02 |
> |   128 |    128 |    6 |   1536 |    3.498 |   219.57 |   10.354 |    74.17 |   13.852 |   110.89 |
> |   128 |    128 |    8 |   2048 |    3.621 |   282.79 |   14.736 |    69.49 |   18.357 |   111.56 |
> |   128 |    128 |   10 |   2560 |    5.542 |   230.95 |   19.563 |    65.43 |   25.106 |   101.97 |
> |   128 |    128 |   12 |   3072 |    5.408 |   284.02 |   24.153 |    63.59 |   29.561 |   103.92 |
> |   128 |    128 |   14 |   3584 |    7.023 |   255.17 |   29.784 |    60.17 |   36.807 |    97.37 |
> |   128 |    128 |   16 |   4096 |    7.103 |   288.33 |   35.599 |    57.53 |   42.702 |    95.92 |
> 
> 
> `llama-batched-bench -m /models1/Qwen_Qwen3-0.6B-Q6_K.gguf -c 4096 -b 512 -ub 512 -ngl 0,1 -npp 128 -ntg 128 -npl 1,2,4,6,8,10,12,14,16 -fa --cache-type-k q8_0`
> 
> |    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
> |-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
> |   128 |    128 |    1 |    256 |    2.136 |    59.93 |    2.950 |    43.39 |    5.086 |    50.34 |
> |   128 |    128 |    2 |    512 |    2.843 |    90.03 |    4.471 |    57.26 |    7.314 |    70.00 |
> |   128 |    128 |    4 |   1024 |    4.506 |   113.62 |    7.563 |    67.70 |   12.069 |    84.85 |
> |   128 |    128 |    6 |   1536 |    7.924 |    96.92 |   11.261 |    68.20 |   19.185 |    80.06 |
> |   128 |    128 |    8 |   2048 |    9.385 |   109.12 |   14.843 |    68.99 |   24.228 |    84.53 |
> |   128 |    128 |   10 |   2560 |   13.274 |    96.43 |   21.822 |    58.66 |   35.096 |    72.94 |
> |   128 |    128 |   12 |   3072 |   14.836 |   103.53 |   30.557 |    50.27 |   45.392 |    67.68 |
> |   128 |    128 |   14 |   3584 |   18.849 |    95.07 |   41.660 |    43.02 |   60.509 |    59.23 |
> |   128 |    128 |   16 |   4096 |   20.788 |    98.52 |   34.703 |    59.01 |   55.492 |    73.81 |
> 
> 
> ### CPU build
> 
> `llama-batched-bench -m /models1/Qwen_Qwen3-0.6B-Q6_K.gguf -c 4096 -b 512 -ub 512 -ngl 0,1 -npp 128 -ntg 128 -npl 1,2,4,6,8,10,12,14,16`
> 
> |    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
> |-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
> |   128 |    128 |    1 |    256 |    0.858 |   149.13 |    3.157 |    40.55 |    4.015 |    63.76 |
> |   128 |    128 |    2 |    512 |    1.683 |   152.13 |    4.879 |    52.47 |    6.562 |    78.02 |
> |   128 |    128 |    4 |   1024 |    3.570 |   143.42 |    7.726 |    66.27 |   11.296 |    90.65 |
> |   128 |    128 |    6 |   1536 |    5.465 |   140.53 |   10.482 |    73.27 |   15.947 |    96.32 |
> |   128 |    128 |    8 |   2048 |    7.761 |   131.94 |   15.193 |    67.40 |   22.954 |    89.22 |
> |   128 |    128 |   10 |   2560 |    9.970 |   128.38 |   19.755 |    64.79 |   29.726 |    86.12 |
> |   128 |    128 |   12 |   3072 |   12.513 |   122.75 |   24.533 |    62.61 |   37.046 |    82.92 |
> |   128 |    128 |   14 |   3584 |   15.011 |   119.38 |   30.032 |    59.67 |   45.043 |    79.57 |
> |   128 |    128 |   16 |   4096 |   17.933 |   114.20 |   35.927 |    57.01 |   53.860 |    76.05 |
> 
> 
> `llama-batched-bench -m /models1/Qwen_Qwen3-0.6B-Q6_K.gguf -c 4096 -b 512 -ub 512 -ngl 0,1 -npp 128 -ntg 128 -npl 1,2,4,6,8,10,12,14,16 -fa --cache-type-k q8_0`
> 
> |    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
> |-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
> |   128 |    128 |    1 |    256 |    1.061 |   120.60 |    3.088 |    41.46 |    4.149 |    61.70 |
> |   128 |    128 |    2 |    512 |    1.668 |   153.51 |    4.754 |    53.85 |    6.422 |    79.73 |
> |   128 |    128 |    4 |   1024 |    3.566 |   143.58 |    7.453 |    68.70 |   11.019 |    92.93 |
> |   128 |    128 |    6 |   1536 |    5.346 |   143.65 |   11.886 |    64.61 |   17.232 |    89.13 |
> |   128 |    128 |    8 |   2048 |    7.491 |   136.70 |   14.897 |    68.74 |   22.388 |    91.48 |
> |   128 |    128 |   10 |   2560 |    9.620 |   133.06 |   22.426 |    57.08 |   32.045 |    79.89 |
> |   128 |    128 |   12 |   3072 |   11.950 |   128.54 |   31.101 |    49.39 |   43.051 |    71.36 |
> |   128 |    128 |   14 |   3584 |   14.372 |   124.69 |   42.149 |    42.52 |   56.520 |    63.41 |
> |   128 |    128 |   16 |   4096 |   17.197 |   119.09 |   34.384 |    59.56 |   51.581 |    79.41 |
> 
> </details>
> 
> 
> @ikawrakow
> 
> > What is the CPU for these graphs?
> 
> [AMD Ryzen 5 3400G](https://www.techpowerup.com/cpu-specs/ryzen-5-3400g.c2204), old and without AVX512 :smile:
> 
> But it's always powered on thus `llama-server` Webui immediately accessible even from phone.
> 
> `
>     Flags:                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxs
>                              r sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl n
>                              onstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_
>                              1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_lega
>                              cy abm sse4a misalignsse 3dnowprefetch osvw skinit wdt tce topoext perfctr_core perfctr
>                              _nb bpext perfctr_llc mwaitx cpb hw_pstate ssbd ibpb vmmcall fsgsbase bmi1 avx2 smep bm
>                              i2 rdseed adx smap clflushopt sha_ni xsaveopt xsavec xgetbv1 clzero xsaveerptr arat npt
>                               lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pft
>                              hreshold avic v_vmsave_vmload vgif overflow_recov succor smca sev sev_es
> `
> 
> 👤 **saood06** replied the **2025-07-07** at **09:18:30**:<br>
> > > I'd be curious about peak batched performance comparisons (I never got around to adding a plot tool to batched-bench)
> > 
> > Results here, click to expand
> 
> Thanks, but not sure what to make of these given you use `-ngl 0,1` (which I think is being interpreted as 1), instead of 99/0 like you did for `sweep-bench`
> 
> Edit:
> 
> >[AMD Ryzen 5 3400G](https://www.techpowerup.com/cpu-specs/ryzen-5-3400g.c2204), old and without AVX512 😄
> 
> My server CPU uses the first CPU architecture with AVX2.
> 
> 👤 **mcm007** replied the **2025-07-07** at **10:43:29**:<br>
> Sorry, `0,1` was meant for `fa` I think. It used 0 in the typo `llm_load_tensors: offloaded 0/29 layers to GPU`.
> 
> <details>
> 
> <summary>CPU/Vulkan/ngl/FA Results</summary>
> 
> ### Vulkan build
> 
> `llama-batched-bench -m /models1/Qwen_Qwen3-0.6B-Q6_K.gguf -c 4096 -b 512 -ub 512 -npp 128 -ntg 128 -npl 1,2,4,6,8,10,12,14,16 -fa -ngl 0`
> 
> |    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
> |-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
> |   128 |    128 |    1 |    256 |    2.027 |    63.14 |    3.037 |    42.15 |    5.064 |    50.55 |
> |   128 |    128 |    2 |    512 |    1.793 |   142.76 |    4.595 |    55.71 |    6.388 |    80.15 |
> |   128 |    128 |    4 |   1024 |    1.839 |   278.46 |    7.841 |    65.30 |    9.679 |   105.79 |
> |   128 |    128 |    6 |   1536 |    3.420 |   224.57 |   14.302 |    53.70 |   17.722 |    86.67 |
> |   128 |    128 |    8 |   2048 |    3.590 |   285.26 |   15.373 |    66.61 |   18.963 |   108.00 |
> |   128 |    128 |   10 |   2560 |    5.156 |   248.23 |   27.476 |    46.59 |   32.633 |    78.45 |
> |   128 |    128 |   12 |   3072 |    5.747 |   267.28 |   41.406 |    37.10 |   47.153 |    65.15 |
> |   128 |    128 |   14 |   3584 |    7.283 |   246.05 |   58.771 |    30.49 |   66.054 |    54.26 |
> |   128 |    128 |   16 |   4096 |    8.226 |   248.97 |   37.488 |    54.63 |   45.714 |    89.60 |
> 
> 
> `llama-batched-bench -m /models1/Qwen_Qwen3-0.6B-Q6_K.gguf -c 4096 -b 512 -ub 512 -npp 128 -ntg 128 -npl 1,2,4,6,8,10,12,14,16 -fa -ngl 99`
> 
> |    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
> |-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
> |   128 |    128 |    1 |    256 |    0.326 |   392.75 |    2.841 |    45.05 |    3.167 |    80.82 |
> |   128 |    128 |    2 |    512 |    0.388 |   660.42 |    3.400 |    75.29 |    3.788 |   135.16 |
> |   128 |    128 |    4 |   1024 |    0.841 |   608.95 |    5.633 |    90.89 |    6.474 |   158.18 |
> |   128 |    128 |    6 |   1536 |    1.328 |   578.33 |    7.383 |   104.03 |    8.711 |   176.34 |
> |   128 |    128 |    8 |   2048 |    1.960 |   522.41 |    9.095 |   112.59 |   11.055 |   185.25 |
> |   128 |    128 |   10 |   2560 |    2.595 |   493.23 |   16.859 |    75.92 |   19.455 |   131.59 |
> |   128 |    128 |   12 |   3072 |    3.487 |   440.48 |   17.976 |    85.45 |   21.463 |   143.13 |
> |   128 |    128 |   14 |   3584 |    4.313 |   415.48 |   19.101 |    93.82 |   23.414 |   153.07 |
> |   128 |    128 |   16 |   4096 |    5.380 |   380.64 |   20.148 |   101.65 |   25.528 |   160.45 |
> 
> 
> `llama-batched-bench -m /models1/Qwen_Qwen3-0.6B-Q6_K.gguf -c 4096 -b 512 -ub 512 -npp 128 -ntg 128 -npl 1,2,4,6,8,10,12,14,16 -ngl 0`
> 
> |    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
> |-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
> |   128 |    128 |    1 |    256 |    2.151 |    59.52 |    3.212 |    39.85 |    5.363 |    47.74 |
> |   128 |    128 |    2 |    512 |    1.815 |   141.07 |    4.438 |    57.69 |    6.252 |    81.89 |
> |   128 |    128 |    4 |   1024 |    1.870 |   273.79 |    7.488 |    68.37 |    9.358 |   109.42 |
> |   128 |    128 |    6 |   1536 |    3.499 |   219.48 |   10.361 |    74.13 |   13.860 |   110.82 |
> |   128 |    128 |    8 |   2048 |    3.622 |   282.70 |   14.533 |    70.46 |   18.155 |   112.81 |
> |   128 |    128 |   10 |   2560 |    5.552 |   230.56 |   19.646 |    65.15 |   25.198 |   101.60 |
> |   128 |    128 |   12 |   3072 |    5.427 |   283.01 |   24.115 |    63.69 |   29.543 |   103.98 |
> |   128 |    128 |   14 |   3584 |    6.983 |   256.63 |   29.911 |    59.91 |   36.894 |    97.14 |
> |   128 |    128 |   16 |   4096 |    7.082 |   289.20 |   36.246 |    56.50 |   43.327 |    94.54 |
> 
> 
> `llama-batched-bench -m /models1/Qwen_Qwen3-0.6B-Q6_K.gguf -c 4096 -b 512 -ub 512 -npp 128 -ntg 128 -npl 1,2,4,6,8,10,12,14,16 -ngl 99`
> 
> |    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
> |-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
> |   128 |    128 |    1 |    256 |    0.303 |   422.98 |    2.686 |    47.65 |    2.989 |    85.65 |
> |   128 |    128 |    2 |    512 |    0.335 |   763.09 |    4.162 |    61.50 |    4.498 |   113.83 |
> |   128 |    128 |    4 |   1024 |    0.679 |   753.86 |    7.281 |    70.32 |    7.960 |   128.65 |
> |   128 |    128 |    6 |   1536 |    1.051 |   730.81 |   10.296 |    74.60 |   11.346 |   135.37 |
> |   128 |    128 |    8 |   2048 |    1.433 |   714.54 |   12.580 |    81.40 |   14.013 |   146.15 |
> |   128 |    128 |   10 |   2560 |    1.855 |   690.11 |   17.271 |    74.11 |   19.126 |   133.85 |
> |   128 |    128 |   12 |   3072 |    2.277 |   674.54 |   18.591 |    82.62 |   20.868 |   147.21 |
> |   128 |    128 |   14 |   3584 |    2.747 |   652.35 |   19.879 |    90.15 |   22.626 |   158.40 |
> |   128 |    128 |   16 |   4096 |    3.213 |   637.39 |   21.080 |    97.15 |   24.293 |   168.61 |
> 
> 
> ### CPU build
> 
> `llama-batched-bench -m /models1/Qwen_Qwen3-0.6B-Q6_K.gguf -c 4096 -b 512 -ub 512 -npp 128 -ntg 128 -npl 1,2,4,6,8,10,12,14,16 -fa`
> 
> |    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
> |-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
> |   128 |    128 |    1 |    256 |    1.079 |   118.58 |    3.090 |    41.43 |    4.169 |    61.40 |
> |   128 |    128 |    2 |    512 |    1.695 |   151.00 |    4.751 |    53.89 |    6.446 |    79.43 |
> |   128 |    128 |    4 |   1024 |    3.609 |   141.89 |    7.772 |    65.88 |   11.380 |    89.98 |
> |   128 |    128 |    6 |   1536 |    5.607 |   136.98 |   15.116 |    50.81 |   20.723 |    74.12 |
> |   128 |    128 |    8 |   2048 |    7.843 |   130.56 |   15.871 |    64.52 |   23.715 |    86.36 |
> |   128 |    128 |   10 |   2560 |   10.113 |   126.57 |   28.216 |    45.36 |   38.329 |    66.79 |
> |   128 |    128 |   12 |   3072 |   12.770 |   120.28 |   42.656 |    36.01 |   55.426 |    55.43 |
> |   128 |    128 |   14 |   3584 |   15.405 |   116.32 |   60.220 |    29.76 |   75.625 |    47.39 |
> |   128 |    128 |   16 |   4096 |   18.308 |   111.86 |   37.814 |    54.16 |   56.122 |    72.98 |
> 
> 
> `llama-batched-bench -m /models1/Qwen_Qwen3-0.6B-Q6_K.gguf -c 4096 -b 512 -ub 512 -npp 128 -ntg 128 -npl 1,2,4,6,8,10,12,14,16`
> 
> |    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
> |-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
> |   128 |    128 |    1 |    256 |    0.891 |   143.70 |    3.195 |    40.07 |    4.085 |    62.66 |
> |   128 |    128 |    2 |    512 |    1.690 |   151.47 |    4.721 |    54.23 |    6.411 |    79.86 |
> |   128 |    128 |    4 |   1024 |    3.582 |   142.94 |    7.592 |    67.44 |   11.174 |    91.64 |
> |   128 |    128 |    6 |   1536 |    5.515 |   139.26 |   10.560 |    72.73 |   16.075 |    95.55 |
> |   128 |    128 |    8 |   2048 |    7.711 |   132.79 |   15.253 |    67.13 |   22.964 |    89.18 |
> |   128 |    128 |   10 |   2560 |    9.933 |   128.87 |   19.750 |    64.81 |   29.682 |    86.25 |
> |   128 |    128 |   12 |   3072 |   12.619 |   121.72 |   24.358 |    63.06 |   36.978 |    83.08 |
> |   128 |    128 |   14 |   3584 |   14.966 |   119.73 |   29.971 |    59.79 |   44.938 |    79.75 |
> |   128 |    128 |   16 |   4096 |   17.959 |   114.04 |   36.230 |    56.53 |   54.189 |    75.59 |
> 
> </details>

---

👤 **firecoperana** replied the **2025-07-06** at **23:37:31**:<br>

You don't need to make the decision so soon. You can wait and see if this improvement in Vulkan draws more interests from Vulkan users or even developers. It's more important for AMD and Intel users, but they may not know about this yet.

---

👤 **Nexesenex** replied the **2025-07-12** at **01:15:49**:<br>

I personally voted against Vulkan, and only because the community's opinion was asked.

@Ikawrakow : My argument would basically go along yours. If there's demand, and most importantly if there's motivation, and even better if there is help, then I'd love to see IKL support Vulkan, because this backend seems to have a future.

But as of now, your development are so valuable on what you master than it might be more pertinent to focus on your art rather than learn a new technique. A technique which could be provided by skilled Vulkan devs to roll in your wheel, rather than to have to do it yourself. Skilled Vulkan devs who might eventually come to IKL and join you, firecoperana and the fray, because IKL is where the good stuff is, quants and big-moe support-wise, and also "welcoming to all good-wills wise".

Just my opinion, I'll be happy whatever you choose.

Especially after the IQ2_KL surprise! :)

---

👤 **gapeleon** replied the **2025-07-17** at **09:04:15**:<br>

I voted 'no' but regret it / can't  remove my vote. I'd rather abstain :)

For me personally, I use this app to get more performance out of my used Nvidia hardware + CPU with MoE's. The biggest win for me would be if someone could improve rpc server performance, as this would make it viable for us to link multiple rigs without cutting prompt processing in half.

But Vulkan would help both Intel and AMD users.
I noticed a lot of people buying multiple MI50's recently to run larger models, and prompt processing on these with Vulkan is incredibly slow.

Intel are releasing a 24GB GPU later this year. And while Openvino and sycl are way faster, there's an issue with Openvino whereby you can't use KV Cache with multiple GPUs. That 48GB dual-GPU one of the board partners is releasing --will effectively be 2x24gb GPUs, so people buying that card would benefit from faster Vulkan performance.

>  I have mixed feelings how to proceed

ik_llama is a passion project right? So perhaps just do what would be most interesting?

> 👤 **ikawrakow** replied the **2025-07-17** at **14:03:38**:<br>
> > ik_llama is a passion project right? So perhaps just do what would be most interesting?
> 
> "Passion" would be pushing it. But yes, it is a hobby project that I started to hack around for fun. It has never been about winning a popularity contest, and I never went out to beat the drum in HN, Reddit, X, etc. But with time quite a few people have found the project useful, and this is what creates the mixed feelings: it is obvious that a high quality Vulkan back-end will be useful for many, I don't need to be convinced of that. At the same time I'm not sure that I will be having fun adding all the `ik_llama.cpp` quants and the optimizations for MoE models to the Vulkan back-end.
> 
> In any case, thank you for voting!
> But 14 votes in total does not provide a very strong motivation.
> 
> 👤 **firecoperana** replied the **2025-07-17** at **15:20:39**:<br>
> It's not a big problem not adding ik_llama.cpp quants and other optimization to vulkan because Vulkan users are accustomed to missing features compared to CUDA, especially if you don't feel like doing it. Back then, there was no IQ quant support, and FA was barely supported in vulkan in mainline until recently, but it does not stop people from using Vulkan. Until there is more interest from Vulkan users, it's fine the way it is now.

---

👤 **FullstackSensei** replied the **2025-07-18** at **00:12:31**:<br>

Found this discussion while searching for references to SYCL to see if building for SYCL is supported (having a lot of compilation errors).
I have two inference rigs powered by Nvidia and I'm re-purposing a dual Cascade Lake machine I have for MoE inference by adding A770s.

I voted for improving the Vulkan backend but here are my two cents:

- This project doesn't get that much attention on reddit, etc compared to llama.cpp. So, he current userbase is a lot smaller. Having this question in the discussions, while appropriate, won't attract that much attention.
- Vulkan is the only backend that's not tied to a specific vendor. Any optimization you make there will be useful on all GPUs, discrete or otherwise. If you can bring Vulkan close to parity with CUDA, it will be a huge win for any device that supports Vulkan, including older GPUs from Nvidia and AMD.
- As firecoperana noted, not all quants need to be supported. A handful of the recent IQs used in recent MoE's like Qwen3-235B, DeepSeek-671B, and Kimi-K2 are more than enough. I'd even argue for supporting only power of two IQ quants only initially to limit scope and effort.
- Inte's A770 is now arguably the cheapest 16GB GPU with decent compute and memory bandwidth, but it doesn't get much attention in the community. Vulkan support would benefit those of us running Arcs, and free us from having to fiddle with OneAPI.

---

👤 **ExeVirus** replied the **2025-07-18** at **02:45:00**:<br>

You are correct to ask this question. Your target users are those with a single powerful GPU and a decent dram CPU combo. 

Those users are power users and small businesses. Further, most serious ones are using 24GB machines or better. They have rocm and cuda, and if Intel ever comes out with a 24GB single card that is actually available, they'll support it properly as well. 

Vulcan helps old hardware, and people that love hassle free setups. I don't think you should be doing that hassle free work yourself, given your users are all very capable of that work/setup, as much as we would like to have that ease of use.

If your goal is mass popularity like llama.cpp, then yeah get started on Vulcan, and also get some help, cause that's a tall order. Just my thoughts

---

👤 **ACWeb23** replied the **2025-07-18** at **04:06:52**:<br>

I think improvements to vulkan performance would be a positive. This would allow uses greater flexibility when deciding on hardware. Also ARC and AMD GPU users would benefit from these improvements.

---

👤 **lin72h** replied the **2025-07-18** at **04:24:40**:<br>

Vote for Vulkan. It's the API that all vendors are pushing hard to support. AMD's RADV driver is really solid, Intel's ANV is steadily improving, and Jeff Bolz  from NVIDIA [has been contributing](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr+author%3Ajeffbolznv) to llama.cpp's Vulkan backend for several months now.

---

👤 **ikawrakow** replied the **2025-07-18** at **04:53:10**:<br>

Wow, I see 18 new votes since I last checked yesterday. For people who came here to vote for Vulkan but are not familiar with this project, the mainline `llama.cpp` Vulkan back-end has been ported to `ik_llama.cpp`(#608), so it should be on par with what you have in mainline. For models utilizing MLA attention (DeepSeek, Kimi-2), `ik_llama.cpp` outperforms `llama.cpp` by quite a margin as it is - see [here](https://github.com/ikawrakow/ik_llama.cpp/pull/608#issuecomment-3069950613).

> 👤 **FullstackSensei** replied the **2025-07-18** at **08:56:51**:<br>
> > Wow, I see 18 new votes since I last checked yesterday. For people who came here to vote for Vulkan but are not familiar with this project, the mainline `llama.cpp` Vulkan back-end has been ported to `ik_llama.cpp`(#608), so it should be on par with what you have in mainline. For models utilizing MLA attention (DeepSeek, Kimi-2), `ik_llama.cpp` outperforms `llama.cpp` by quite a margin as it is - see [here](https://github.com/ikawrakow/ik_llama.cpp/pull/608#issuecomment-3069950613).
> 
> I took the liberty of posting about this discussion on LocalLLaMA and IntelArc subreddits. Hope you don't mind! Your work makes large models like DeepSeek and Kimi usable on hardware that doesn't cost a kidney, and Vulkan optimizations would only lower the cost to run such models at decent speeds.
> 
> This project doesn't get the exposure it deserves, IMO.. So, I thought at worst more people will become familiar with it.
> 
> 👤 **ikawrakow** replied the **2025-07-18** at **11:59:25**:<br>
> > I took the liberty of posting about this discussion on LocalLLaMA and IntelArc subreddits. Hope you don't mind! 
> 
> This project was the best kept secret on Github for a while, but it no longer is, so feel free to post about it.
> 
> > This project doesn't get the exposure it deserves, IMO
> 
> Thank you.

---

👤 **DealsBeam** replied the **2025-07-18** at **11:54:36**:<br>

Intel Arc GPUs would greatly benefit from Vulkan improvement, thanks for your hard work and dedicating your time on this great project.

> 👤 **ikawrakow** replied the **2025-07-18** at **12:00:32**:<br>
> > Intel Arc GPUs would greatly benefit from Vulkan improvement
> 
> My understanding was that the `llama.cpp` SYCL backend was the better option for Intel GPUs. This is no longer the case?