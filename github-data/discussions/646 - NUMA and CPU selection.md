### [Discussion #646](https://github.com/ikawrakow/ik_llama.cpp/discussions/646) - NUMA and CPU selection

| **Author** | `joshuakoh1` |
| :--- | :--- |
| **Created** | 2025-07-25 |
| **Updated** | 2025-07-25 |

---

#### Description

Hello,

A great fan of this repo. Main llama.cpp was definitely getting too bloated.

Currently shopping for new CPU and would like to clarify some crucial information. I've had bad experiences with NUMA so far, moving from single socket EPYC to dual and back to single. 

My understanding from my research so far is that the higher core count EPYC CPUs will run into NUMA issues even for single socket as CCD count grows. 

My primary use case is to load MOE models like Kimi and Deepseek and my understanding is that there is still no way to bind specific experts per GPU/NUMA domain.

Am I right to say that I should be avoiding the higher CCD CPUs like the 9654/9754 for the foreseeable future?

Thanks!

---

#### 🗣️ Discussion

👤 **ikawrakow** commented on **2025-07-25** at **05:30:21**

NUMA is a topic that I want to do something about, but nothing has been done at this point.

I'm not sure if the high core-count CPU's can be configured as a single NUMA node. Can one not simply try in a cloud instance?

Here is a [comment](https://github.com/ikawrakow/ik_llama.cpp/issues/629#issuecomment-3092549546) from someone using a 9355 EPYC (12 memory channels) and getting a very decent CPU-only TG performance with it.

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

---

👤 **ikawrakow** commented on **2025-07-25** at **09:07:41**

@joshuakoh1 One more thing: apart from any single vs dual socket considerations, if it was me, I would select a CPU from the 9005 series rather than the 9004 series for 2 reasons:
* Theoretically higher memory bandwidth
* Real 512-bit instructions

The 1st point may or may not be important as we are not able to come even close to saturating the 9004 theoretical memory bandwidth during TG with the big MoE models, but in case we figure out where the bottleneck is, the 9005 series will give a better TG performance (independently of single vs dual socket, NUMA, number of CCDs, etc.)

The second point is definitely important. The 9004 series uses the Zen4 core, which has a fairly comprehensive `AVX512` support, but 512-bit instructions are implemented as two 256-instructions, so there is little to no gain from using `AVX512`<sup>1</sup>. The Zen5 core used in the 9005 series is the first AMD core to have real 512-bit instructions. The effect on PP performance can be very significant. For instance, for prompt processing, in `ik_llama.cpp` quantized tensors get repacked to `Q8_0_R8` or `Q8_K_R8` before performing matrix multiplications (GEMM). The GEMM implementation for `Q8_0_R8` uses `AVX512`, while GEMM for `Q8_K_R8` does not. On my Ryzen-7950X CPU (Zen4 core), `Q8_K_R8` GEMM is ~20% faster than `Q8_0_R8` GEMM (just because `Q8_K_R8` needs fewer conversions to float and multiplications with block scales). But on @ubergarm's Ryzen-9950X CPU it is the other way around. PR #610 has a POC for `Q8_K_R8` GEMM with `AVX512`. With this PR, the 16-core Ryzen-9950X beats the 24-core Ryzen-7965WX, so that will translate in more than 50% better PP performance per core/clock for the 9005 series EPYC. 

---
<sup>1</sup> There are some relatively minor performance gains when one can make use of the additional `AVX512` instructions, particularly instructions with a mask. But for the bulk of the computation, which is performing `int8_t` dot products between SIMD vectors, there is no benefit from `AVX512` with the Zen4 core.

> 👤 **sousekd** replied on **2025-07-25** at **09:45:02**
> 
> > The Zen5 core used in the 9005 series is the first AMD core to have real 512-bit instructions. The effect on PP performance can be very significant.
> 
> FYI @ikawrakow: I've been trying to validate this by testing #610 on @ubergarm's IQ5_K [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/ubergarm/Qwen3-235B-A22B-Instruct-2507-GGUF) on a Zen 5 EPYC, but I haven’t been able to achieve any measurable improvements so far. There's a BIOS switch to explicitly enable 512-bit AVX512 instructions (instead of “auto,” which might default to 2×256-bit), but enabling it didn't help. I’m not sure what role Windows might be playing here (runtime or when compiling), but I should have Linux on this server in a couple of weeks, so the Windows factor will finally be out of the equation for all my testing.

> 👤 **sousekd** replied on **2025-07-25** at **11:19:47**
> 
> > Would a 16 core 9115 make more sense than a 32/48 core 9354/9454?
> 
> I think the 9115 would be a missed opportunity for significantly better performance. The 9355 costs only about $1500 more, while the bulk of the investment still goes into RAM — and the motherboards aren't cheap either. Going with the previous generation would certainly be more budget-friendly overall. Personally, seeing how unexpectedly well it performs, I now wish I had gone with the EPYC 9555 instead of the 9355, and 1152 GB of RAM instead of 768, just to be able to run even larger models.