## 🗣️ [Discussion #646](https://github.com/ikawrakow/ik_llama.cpp/discussions/646) - NUMA and CPU selection

| **Author** | `joshuakoh1` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-07-25 |
| **Updated** | 2025-07-27 |

---

## 📄 Description

Hello,

A great fan of this repo. Main llama.cpp was definitely getting too bloated.

Currently shopping for new CPU and would like to clarify some crucial information. I've had bad experiences with NUMA so far, moving from single socket EPYC to dual and back to single. 

My understanding from my research so far is that the higher core count EPYC CPUs will run into NUMA issues even for single socket as CCD count grows. 

My primary use case is to load MOE models like Kimi and Deepseek and my understanding is that there is still no way to bind specific experts per GPU/NUMA domain.

Am I right to say that I should be avoiding the higher CCD CPUs like the 9654/9754 for the foreseeable future?

Thanks!

---

## 💬 Discussion

👤 **ikawrakow** commented on **2025-07-25** at **05:30:21**

NUMA is a topic that I want to do something about, but nothing has been done at this point.

I'm not sure if the high core-count CPU's can be configured as a single NUMA node. Can one not simply try in a cloud instance?

Here is a [comment](https://github.com/ikawrakow/ik_llama.cpp/issues/629#issuecomment-3092549546) from someone using a 9355 EPYC (12 memory channels) and getting a very decent CPU-only TG performance with it.

> 👤 **saood06** replied on **2025-07-25** at **05:37:39**
> 
> >I'm not sure if the high core-count CPU's can be configured as a single NUMA node. Can one not simply try in a cloud instance?
> 
> I've seen BIOS's that mirror memory to create one NUMA node. (Not relevant on gear I have, and not sure if present on gear I have ).

> 👤 **saood06** replied on **2025-07-25** at **05:42:05**
> 
> >NUMA is a topic that I want to do something about, but nothing has been done at this point.
> 
> Someone on the reddit thread about this repo's incident commented this:
> 
> >I solved the llama-cpp bug where multiple numas made things slower, not faster (everything was being bound to a single numa node due to the first-touch linux kernel policy with malloc... everything was being zeroed from the main thread so it always ended up bound to the main thread's numa). Now it's numa aware and buffers should be bound to the thread-local numa. So each numa will only have to reach across the link some of the time for matmuls. Instead of the main thread numa having full local access and all remote numas reaching across the bus 100% of the time as now.
> 
> And in another comment:
> 
> >The problem with the existing code was that numactl was not really effective since all the buffers were initialised on the main thread and would all get pinned to the main thread's numa by the kernel no matter what you set in numactl. So that was broken anyways. You'd end up with all of the model in one numa but the threads all split across both numas, so it was very ridiculous.

---

👤 **ikawrakow** commented on **2025-07-25** at **06:01:14**

I haven't seen the Reddit thread, but what I would try first when I get access to a NUMA system is to delay tensor data loading until the warm-up graph computation, and there have each thread load the tensor portions that they will be working on. One could also multi-thread tensor data loading, but one needs to make sure that the correct tensor portions are loaded by the respective threads, which happens automatically if data loading is done within the warm-up graph computation. I have done something along these lines in the past at a $former_job in the context of a large-scale optimization problem where the system matrix had to be distributed that way to (nearly) double the performance on the dual-socket production system. I haven't tried any of it yet because I want to test on a real NUMA system. 

This and the Vulkan back-end are competing for being the next thing to focus on when I come back from vacation.

---

👤 **sousekd** commented on **2025-07-25** at **07:02:21**

> Am I right to say that I should be avoiding the higher CCD CPUs like the 9654/9754 for the foreseeable future?

This is a great topic I've been trying to understand too - and the same question I asked myself before shopping. I've seen people on the Level1Techs forum blaming low CCD count for poor memory bandwidth, with some suggesting the EPYC 9175F as a great CPU because it has 16 CCDs (1 per core).

But then I came across a paper explaining cross-CCD memory latency and bandwidth, and another with memory bandwidth benchmarks for various 9004/9005 CPUs. I can't access them now for some reason (everything I touch gets nuked lately 😅), but try googling "Fujitsu Genoa Turin memory performance white paper" - you might find it.

[Here's a chart](https://www.thomas-krenn.com/en/wiki/AMD_EPYC_9005_Turin) of all AMD EPYC 9005 Turin CPUs I found extremely useful.

> 👤 **joshuakoh1** replied on **2025-07-25** at **09:47:38**
> 
> The 9175F or F series chips certainly look interesting IF the software is mature enough for CCD pinning. Not sure if this is the case or will be the case in the near future. As far as I've found out, seems like all cross CCD communication will go across the infinity fabric which would end up being a massive bottleneck if not properly managed.

---

👤 **ikawrakow** commented on **2025-07-25** at **09:07:41**

@joshuakoh1 One more thing: apart from any single vs dual socket considerations, if it was me, I would select a CPU from the 9005 series rather than the 9004 series for 2 reasons:
* Theoretically higher memory bandwidth
* Real 512-bit instructions

The 1st point may or may not be important as we are not able to come even close to saturating the 9004 theoretical memory bandwidth during TG with the big MoE models, but in case we figure out where the bottleneck is, the 9005 series will give a better TG performance (independently of single vs dual socket, NUMA, number of CCDs, etc.)

The second point is definitely important. The 9004 series uses the Zen4 core, which has a fairly comprehensive `AVX512` support, but 512-bit instructions are implemented as two 256-instructions, so there is little to no gain from using `AVX512`<sup>1</sup>. The Zen5 core used in the 9005 series is the first AMD core to have real 512-bit instructions. The effect on PP performance can be very significant. For instance, for prompt processing, in `ik_llama.cpp` quantized tensors get repacked to `Q8_0_R8` or `Q8_K_R8` before performing matrix multiplications (GEMM). The GEMM implementation for `Q8_0_R8` uses `AVX512`, while GEMM for `Q8_K_R8` does not. On my Ryzen-7950X CPU (Zen4 core), `Q8_K_R8` GEMM is ~20% faster than `Q8_0_R8` GEMM (just because `Q8_K_R8` needs fewer conversions to float and multiplications with block scales). But on @ubergarm's Ryzen-9950X CPU it is the other way around. PR [#610](https://github.com/ikawrakow/ik_llama.cpp/issues/610) has a POC for `Q8_K_R8` GEMM with `AVX512`. With this PR, the 16-core Ryzen-9950X beats the 24-core Ryzen-7965WX, so that will translate in more than 50% better PP performance per core/clock for the 9005 series EPYC. 

---
<sup>1</sup> There are some relatively minor performance gains when one can make use of the additional `AVX512` instructions, particularly instructions with a mask. But for the bulk of the computation, which is performing `int8_t` dot products between SIMD vectors, there is no benefit from `AVX512` with the Zen4 core.

> 👤 **joshuakoh1** replied on **2025-07-25** at **09:43:59**
> 
> Regarding your second point, would a 16 core 9115 make more sense than a 32/48 core 9354/9454?

> 👤 **sousekd** replied on **2025-07-25** at **09:45:02**
> 
> > The Zen5 core used in the 9005 series is the first AMD core to have real 512-bit instructions. The effect on PP performance can be very significant.
> 
> FYI @ikawrakow: I've been trying to validate this by testing [#610](https://github.com/ikawrakow/ik_llama.cpp/issues/610) on @ubergarm's IQ5_K [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/ubergarm/Qwen3-235B-A22B-Instruct-2507-GGUF) on a Zen 5 EPYC, but I haven’t been able to achieve any measurable improvements so far. There's a BIOS switch to explicitly enable 512-bit AVX512 instructions (instead of “auto,” which might default to 2×256-bit), but enabling it didn't help. I’m not sure what role Windows might be playing here (runtime or when compiling), but I should have Linux on this server in a couple of weeks, so the Windows factor will finally be out of the equation for all my testing.

> 👤 **sousekd** replied on **2025-07-25** at **11:19:47**
> 
> > Would a 16 core 9115 make more sense than a 32/48 core 9354/9454?
> 
> I think the 9115 would be a missed opportunity for significantly better performance. The 9355 costs only about $1500 more, while the bulk of the investment still goes into RAM — and the motherboards aren't cheap either. Going with the previous generation would certainly be more budget-friendly overall. Personally, seeing how unexpectedly well it performs, I now wish I had gone with the EPYC 9555 instead of the 9355, and 1152 GB of RAM instead of 768, just to be able to run even larger models.

> 👤 **ubergarm** replied on **2025-07-26** at **19:04:13**
> 
> @sousekd 
> 
> > I've been trying to validate this by testing https://github.com/ikawrakow/ik_llama.cpp/pull/610 on @ubergarm's IQ5_K [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/ubergarm/Qwen3-235B-A22B-Instruct-2507-GGUF) on a Zen 5 EPYC, but I haven’t been able to achieve any measurable improvements so far.
> 
> I just ran some A/B comparisons for exact same setup and three of my Qwen3-235B-A22B-Thinking-2507-GGUF quants. This is CPU-only backend. I can still see some uplift especially at lower kv-cache length. I haven't tested at different batch sizes though. I know on my local amd 9950x the avx_vnni realy helps PP. Will need more folks doing llama-sweep-bench across different CPUs to see where it makes the most benefits or not.
> 
> *EDIT* this was actually on NUMA0 which is down a bank of 64GB RAM so slightly less bandwidth.
> 
> <details>
> 
> <summary>👈 command and data</summary>
> 
> ```bash
> #model=/mnt/raid/hf/Qwen3-235B-A22B-Thinking-2507-GGUF/IQ2_KL/Qwen3-235B-A22B-Thinking-2507-IQ2_KL-00001-of-00002.gguf
> model=/mnt/raid/hf/Qwen3-235B-A22B-Thinking-2507-GGUF/IQ4_KSS/Qwen3-235B-A22B-Thinking-2507-IQ4_KSS-00001-of-00003.gguf
> #model=/mnt/raid/hf/Qwen3-235B-A22B-Thinking-2507-GGUF/IQ5_K/Qwen3-235B-A22B-Thinking-2507-IQ5_K-00001-of-00004.gguf
> 
> numactl -N 0 -m 0 \
> ./build/bin/llama-sweep-bench \
>     --model "$model"\
>     --ctx-size 20480 \
>     -ctk q8_0 -ctv q8_0 \
>     -fa -fmoe \
>     -ub 4096 -b 4096 \
>     --threads 128 \
>     --threads-batch 192 \
>     --numa numactl \
>     --warmup-batch \
>     --no-mmap
> ```
> ## main+ik/q8_k_r8_avx512 IQ5_K 161.722 GiB (5.909 BPW) PPL=4.2213
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |  4096 |   1024 |      0 |   13.487 |   303.71 |   83.981 |    12.19 |
> |  4096 |   1024 |   4096 |   16.210 |   252.68 |   89.762 |    11.41 |
> |  4096 |   1024 |   8192 |   21.058 |   194.51 |   95.473 |    10.73 |
> |  4096 |   1024 |  12288 |   23.931 |   171.16 |  101.712 |    10.07 |
> |  4096 |   1024 |  16384 |   26.093 |   156.98 |  105.020 |     9.75 |
> 
> ## main+ik/q8_k_r8_avx512 IQ4_KSS 113.368 GiB (4.142 BPW) PPL=4.2799
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |  4096 |   1024 |      0 |   13.086 |   313.00 |   70.662 |    14.49 |
> |  4096 |   1024 |   4096 |   15.590 |   262.74 |   77.588 |    13.20 |
> |  4096 |   1024 |   8192 |   19.053 |   214.98 |   82.916 |    12.35 |
> |  4096 |   1024 |  12288 |   22.733 |   180.18 |   89.105 |    11.49 |
> |  4096 |   1024 |  16384 |   25.114 |   163.09 |   93.027 |    11.01 |
> 
> ## main+ik/q8_k_r8_avx512 IQ2_KL 81.866 GiB (2.991 BPW) PPL=4.6608
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |  4096 |   1024 |      0 |   13.575 |   301.74 |   67.022 |    15.28 |
> |  4096 |   1024 |   4096 |   16.507 |   248.13 |   73.716 |    13.89 |
> |  4096 |   1024 |   8192 |   19.768 |   207.20 |   78.186 |    13.10 |
> |  4096 |   1024 |  12288 |   22.374 |   183.07 |   82.825 |    12.36 |
> |  4096 |   1024 |  16384 |   25.316 |   161.79 |   86.058 |    11.90 |
> 
> ## main@4e9c78c0 IQ5_K 161.722 GiB (5.909 BPW) PPL=4.2213
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |  4096 |   1024 |      0 |   14.998 |   273.10 |   83.933 |    12.20 |
> |  4096 |   1024 |   4096 |   16.555 |   247.41 |   90.759 |    11.28 |
> |  4096 |   1024 |   8192 |   20.488 |   199.92 |   96.429 |    10.62 |
> |  4096 |   1024 |  12288 |   23.990 |   170.74 |  102.942 |     9.95 |
> |  4096 |   1024 |  16384 |   26.551 |   154.27 |  104.963 |     9.76 |
> 
> ## main@4e9c78c0 IQ4_KSS 113.368 GiB (4.142 BPW) PPL=4.2799
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |  4096 |   1024 |      0 |   14.781 |   277.11 |   72.622 |    14.10 |
> |  4096 |   1024 |   4096 |   17.259 |   237.33 |   79.266 |    12.92 |
> |  4096 |   1024 |   8192 |   19.752 |   207.38 |   84.637 |    12.10 |
> |  4096 |   1024 |  12288 |   23.824 |   171.92 |   90.505 |    11.31 |
> |  4096 |   1024 |  16384 |   26.772 |   152.99 |   95.708 |    10.70 |
> 
> ## main@4e9c78c0 IQ2_KL 81.866 GiB (2.991 BPW) PPL=4.6608
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |  4096 |   1024 |      0 |   14.805 |   276.66 |   70.098 |    14.61 |
> |  4096 |   1024 |   4096 |   17.646 |   232.13 |   76.985 |    13.30 |
> |  4096 |   1024 |   8192 |   20.264 |   202.13 |   82.384 |    12.43 |
> |  4096 |   1024 |  12288 |   22.525 |   181.84 |   87.735 |    11.67 |
> |  4096 |   1024 |  16384 |   26.082 |   157.05 |   91.991 |    11.13 |
> 
> </details>
> 
> <img width="2087" height="1141" alt="sweep-bench-avx512-testing-qwen3-235b-a22b-thinking" src="https://github.com/user-attachments/assets/1e633af2-53fe-4138-bf3d-a46b1450dcf6" />

> 👤 **sousekd** replied on **2025-07-27** at **08:01:31**
> 
> > I can still see some uplift especially at lower kv-cache length.
> 
> @ubergarm Do you build/compile with some unusual params or just the common set? I would like to reproduce your gains.

> 👤 **ubergarm** replied on **2025-07-27** at **16:01:08**
> 
> @sousekd 
> 
> > Do you build/compile with some unusual params or just the common set? I would like to reproduce your gains.
> 
> Nothing special no.
> 
> ```
> cd ik_llama.cpp
> git checkout main
> git pull
> git checkout ik/q8_k_r8_avx512
> git checkout -b testing
> git rebase main
> 
> # build, if you're using GPU set -DGGML_CUDA=1 -DGGML_SCHED_MAX_COPIES=1
> cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=0 -DGGML_BLAS=0 -DGGML_VULKAN=0
> cmake --build build --config Release -j $(nproc)
> 
> # when you're all done
> git checkout main
> git branch -D testing
> ```

> 👤 **ikawrakow** replied on **2025-07-27** at **16:08:10**
> 
> I think on Windows one needs to explicitly turn on the required `AVX512` features. On Linux this happens automatically  with `-DGGML_NATIVE=ON` (it is already on by default), but not on Windows. The required CPU features for the fast path are `AVX512F, AVX512VNNI, AVX512VL, AVX512BW, AVX512DQ`. I don't think there are `cmake` variables for all of these, so one probably needs to work with `-DGGML_ARCH_FLAGS=...` (fill in whatever is required for the compiler used for building).

> 👤 **sousekd** replied on **2025-07-27** at **16:21:12**
> 
> @ubergarm @ikawrakow Yeah that's what I do. I tested various configurations with your Kimi-K2 and most recent Qwen quants today but have not been able to achieve any improvements with AVX512. But I test with GPU in the mix, so the last thing I'll try, just for the sake of testing it, is a build without CUDA.

> 👤 **ikawrakow** replied on **2025-07-27** at **16:35:07**
> 
> > Yeah that's what I do.
> 
> Which likely means that you are not using `AVX512` at all, or at least not using the fast path that @ubergarm uses on Linux, and this is why the PR in question has no impact for you.

> 👤 **sousekd** replied on **2025-07-27** at **18:42:40**
> 
> > Which likely means that you are not using `AVX512` at all, or at least not using the fast path that @ubergarm uses on Linux, and this is why the PR in question has no impact for you.
> 
> It is absolutely possible (and likely) I am doing something wrong. I finally posted some results [here](https://github.com/ikawrakow/ik_llama.cpp/pull/610#issuecomment-3124618208), suggesting minor improvements when running CPU only. My build was:
> 
> ```
> cmake -B build -G Ninja `
>   -DCMAKE_BUILD_TYPE=Release `
>   -DCMAKE_C_COMPILER="clang-cl" `
>   -DCMAKE_CXX_COMPILER="clang-cl" `
>   -DGGML_CUDA=OFF `
>   -DGGML_AVX512=ON `
>   -DGGML_AVX512_VNNI=ON `
>   -DGGML_AVX512_VBMI=ON `
>   -DGGML_AVX512_BF16=ON `
>   -DGGML_BLAS=OFF `
>   -DGGML_CCACHE=OFF `
>   -DCMAKE_C_FLAGS='/clang:-march=znver5' `
>   -DCMAKE_CXX_FLAGS='/EHsc /clang:-march=znver5' `
>   -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON `
>   -DLLAMA_CURL=OFF `
>   -DBUILD_SHARED_LIBS=OFF
> ```