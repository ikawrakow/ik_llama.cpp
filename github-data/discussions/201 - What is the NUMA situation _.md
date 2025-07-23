### 🗣️ [#201](https://github.com/ikawrakow/ik_llama.cpp/discussions/201) - What is the NUMA situation ?

| **Author** | `bhugueney` |
| :--- | :--- |
| **Created** | 2025-02-11 |
| **Updated** | 2025-05-21 |

---

#### Description

It seems to me that output generation being memory bandwidth bounded and LLM requiring a lot of RAM , a cheap way to try increase both RAM amount and bandwidth is to go for NUMA.
For instance, a dual Epyc server can have 16 or 24 memory channels each CPU can also have up to 4 NUMA domains for best theoretical performance (also, on Gen 2 Epyc at least, L3 cache is shared only amongst cores on the same CCX).
However, there are many pitfalls to efficient NUMA programming especially to minimize cross NUMA domain memory and PCIe access.

It is my understanding that llama.cpp is trying to avoid the most basic problems (e.g. allocation everything in 1 NUMA domain) but more work needs to be done.
[KTransformers](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md#some-explanations) just duplicates matrices on each NUMA domain !

[vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/index.html#other-considerations) can do tensor parallelism on NUMA :  «In general each NUMA node is treated as one GPU card. »

Is ik_llama.cpp NUMA aware ? If not, are there plans to make it NUMA aware ?
Thx !

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2025-02-11** at **06:09:03**:<br>

In `ik_llama.cpp`, being a fork of `llama.cpp`, the NUMA situation is the same as in `llama.cpp`.

Improving performance on NUMA systems is something I would be interested in looking into, but I don't have a dual socket system available (with enough memory bandwidth to make it interesting), and I'm just a lonely guy hacking here for fun without the resources to go and rent/buy such a system.

> 👤 **bhugueney** replied the **2025-02-11** at **10:56:00**:<br>
> Thx !
> I sure hope my message didn't come of as complaining : I've very grateful for what you already did !
> If you are interested I will try to provide you full access to my dual Epyc server with 16 × 64 GB of DDR4 @3200.
> 
> 👤 **ikawrakow** replied the **2025-02-11** at **14:47:10**:<br>
> This would be of course great, but I'm hesitant to promise to tackle the NUMA issue right away. 
> 
> When you say "full access", you mean you are not going to be using the system while I'm using it? Which Epycs do you have?
> 
> 👤 **bhugueney** replied the **2025-02-11** at **23:17:06**:<br>
> I'm not expecting any promises, especially as I'm afraid llama.cpp cannot be patched to become NUMA efficient. My (very) limited understanding is that people ran llama.cpp CPU backend on NUMA and got bad performance because one thread was doing all the memory allocation (so in one NUMA domain) and they started trying to address that by patching the CPU backend. Unfortunately, such approach seems doomed to hit a wall as NUMA efficiency probably requires a different architecture more like a multi-GPU backend with tensor parallelism where each NUMA domain would be treated like a GPU wrt trying to minimize inter GPU communication and maximize parallelism. This is the vLLM approach for NUMA if I'm note mistaken.
> 
> When I say "full access", I mean IPMI access while I'm not using it. But I have to figure things out first. Epycs would be 7R32 (same as AWS c5a instances).
> 
> 👤 **saood06** replied the **2025-02-11** at **23:58:26**:<br>
> So in regards to the current state of llama.cpp/ik_llama.cpp NUMA performance I don't think it's that bad. I've seen a few reports from a few users on more modern NUMA machines than mine report performance running multiple instances of llama.cpp on each NUMA domain isolated, vs running one larger instance on all NUMA domains and although there was gain to be had it wasn't that dramatic of a difference. My older NUMA machine also gets decent performance for it's bandwidth.
> 
> I'm looking into expert parallelism for the Deepseek V3/R1 MoE model, which should benefit NUMA systems. The plan for that is port over the PR which allows you to specify what tensor is loaded onto what backend, change the tensor representation of this model to not consolidate the experts. At that point I'd test performance with that and each NUMA node on a separate RPC backend, since changing ik_llama.cpp to create a backend for each NUMA domain might require a lot more work, but I'd look into it once I get there.

---

👤 **saood06** replied the **2025-03-13** at **05:53:54**:<br>

There is actually a good discussion on mainline: https://github.com/ggml-org/llama.cpp/discussions/12088

They did test ik_llama.cpp (but in only with a single NUMA Node on a single CPU at Q8_0) where it still outperformed mainline for CPU only.

Also you can look at zts9989's comment [here](https://github.com/ggml-org/llama.cpp/pull/11397#issuecomment-2716225570)  where he talks about NUMA and what llama.cpp could improve on after he found that "approximately 50% of CPU usage is spent on thread synchronization" when running Deepseek R1 with multiple numa nodes.

> 👤 **ikawrakow** replied the **2025-03-13** at **07:27:34**:<br>
> > They did test ik_llama.cpp (but in only with a single NUMA Node on a single CPU at Q8_0) where it still outperformed mainline for CPU only.
> 
> Where can I find the test results?
> 
> 👤 **saood06** replied the **2025-03-13** at **07:44:42**:<br>
> In the linked post the second table under 6980P Benchmarks has it, but pasting it here for reference:
> 
> Quantization | Tokens/Second | NUMA Configuration
> -- | -- | --
> Q8_0 | 6.6 | 1x NUMA Node on 1x CPU ik_llama
> Q8_0 | 6.2 | 1x NUMA Node on 1x CPU
> 
> This is the only published result for ik_llama but they do state "Keep an eye on [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) fork which has interesting optimizations." so they may run more.
> 
> 👤 **saood06** replied the **2025-03-13** at **08:45:24**:<br>
> I forgot he had much more detailed results under Methodology and Notes, there is a section for ik_llama.cpp showing the command and bench numbers, interestingly ik_llama.cpp performance peaked at 128 threads for both PP and TG compared to peaking at 86 threads for TG and 128 threads for PP in mainline. He also shares PP numbers as well, where ik_llama again shows better performance than mainline. He does explicitly state TODO for testing ik_llama.cpp for 2x CPU Q8_0.
> 
> Again pasting the segment of his post featuring ik_llama.cpp for reference:
> 
> <div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code class="notranslate">numactl -N 0 -m 0 \
>     ./build/bin/llama-bench \
>     --model /mnt/ai/models/unsloth/DeepSeek-R1-GGUF/DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00001-of-00015.gguf \
>     --cache-type-k f16 \
>     --cache-type-v f16 \
>     --numa numactl \
>     --threads 64,43,64,86,128,172
> </code></pre><div class="zeroclipboard-container position-absolute right-0 top-0">
>     
>   </div></div>
> <p dir="auto"><strong>Results</strong></p>
> 
> model | size | params | backend | threads | test | t/s
> -- | -- | -- | -- | -- | -- | --
> deepseek2 671B Q8_0 | 664.29 GiB | 671.03 B | CPU | 64 | pp512 | 56.86 ± 7.21
> deepseek2 671B Q8_0 | 664.29 GiB | 671.03 B | CPU | 64 | tg128 | 4.86 ± 0.01
> deepseek2 671B Q8_0 | 664.29 GiB | 671.03 B | CPU | 43 | pp512 | 40.62 ± 0.02
> deepseek2 671B Q8_0 | 664.29 GiB | 671.03 B | CPU | 43 | tg128 | 3.69 ± 0.00
> deepseek2 671B Q8_0 | 664.29 GiB | 671.03 B | CPU | 64 | pp512 | 57.67 ± 4.62
> deepseek2 671B Q8_0 | 664.29 GiB | 671.03 B | CPU | 64 | tg128 | 4.89 ± 0.00
> deepseek2 671B Q8_0 | 664.29 GiB | 671.03 B | CPU | 86 | pp512 | 62.21 ± 13.63
> deepseek2 671B Q8_0 | 664.29 GiB | 671.03 B | CPU | 86 | tg128 | 5.69 ± 0.00
> deepseek2 671B Q8_0 | 664.29 GiB | 671.03 B | CPU | 128 | pp512 | 78.89 ± 21.46
> deepseek2 671B Q8_0 | 664.29 GiB | 671.03 B | CPU | 128 | tg128 | 6.60 ± 0.00
> deepseek2 671B Q8_0 | 664.29 GiB | 671.03 B | CPU | 172 | pp512 | 70.63 ± 0.58
> deepseek2 671B Q8_0 | 664.29 GiB | 671.03 B | CPU | 172 | tg128 | 5.05 ± 0.00

---

👤 **ikawrakow** replied the **2025-03-13** at **11:55:55**:<br>

@saood06 

Thanks for alerting me to this thread.

They have tested the lowest performing configuration in https://github.com/ggml-org/llama.cpp/discussions/12088 (but this is also to be expected as I don't have any documentation on the new features, so one needs to go through the PRs to discover them).  

For instance, here is a table for DeepSeek-Lite `pp512` performance on my Ryzen-7950X using `Q8_0`. The first row is the configuration used in https://github.com/ggml-org/llama.cpp/discussions/12088, the last is the best possible result for `pp512`.  There is a 50% difference, so I wouldn't be surprised if it is possible to get 100+ t/s on their test system considering the 78 t/s they got with the vanilla settings.

| model               | threads | fa | rtr | fmoe |          test |              t/s |
| ------------------- | ------: | -: | --: | ---: | ------------: | ---------------: |
| deepseek2 16B Q8_0  |      16 |  0 |   0 |    0 |         pp512 |    433.04 ± 1.44 |
| deepseek2 16B Q8_0  |      16 |  1 |   0 |    0 |         pp512 |    440.25 ± 2.54 |
| deepseek2 16B Q8_0  |      16 |  0 |   0 |    1 |         pp512 |    441.58 ± 3.34 |
| deepseek2 16B Q8_0  |      16 |  1 |   0 |    1 |         pp512 |    452.19 ± 1.21 |
| deepseek2 16B Q8_0  |      16 |  0 |   1 |    0 |         pp512 |    607.32 ± 5.09 |
| deepseek2 16B Q8_0  |      16 |  1 |   1 |    0 |         pp512 |    625.10 ± 7.66 |
| deepseek2 16B Q8_0  |      16 |  0 |   1 |    1 |         pp512 |    627.87 ± 4.54 |
| deepseek2 16B Q8_0  |      16 |  1 |   1 |    1 |         pp512 |    652.81 ± 3.52 |

TG is a very different story. There performance is clearly dominated by memory access patterns and thread synchronization, and I cannot look into optimizing this aspect without having access to such a system. As it stands, the achieved performance is nowhere near the maximum theoretical performance. The tested 6980P has a theoretical bandwidth of 512? GiB/s, so 8X my Ryzen-7950X. I get `tg128=22.3 t/s` for `Q8_0`, DeepSeek-Lite has ~15X fewer active parameters, so per napkin math we expect `8*22.3/15 = 11.9 t/s`, so nearly 2X of what is being measured. In contrast, the 22.3 t/s for the `Q8_0` quantized DeepSeek-Lite on my Ryzen-7950X correspond to fetching model weights at a rate of about 57 GiB/s, so pretty close to the theoretical maximum (and I have never seen anything more than 60 GiB/s on the Ryzen-7950X, even for dense models, which is probably due to the few percent synchronization overhead).  

@ubergarm

Very interesting results, thank you for posting and including my little LLM inference playground in the results. I have seen a higher than usual amount of stars added to my repository in the last few days, I guess this must be due to your post. 

I'm curious which `AVX512` extensions are supported by this CPU to understand if vanilla `AVX2` is being used, or the code optimized for the Zen4 core (requires `AVX512F, AVX512VNNI, AVX512VL, AVX512BW, AVX512DQ`).

Playing with some of the more advanced options that mainline `llama.cpp` does not have would be of course very interesting too.

> 👤 **saood06** replied the **2025-03-13** at **21:20:04**:<br>
> >I'm curious which AVX512 extensions are supported by this CPU to understand if vanilla AVX2 is being used, or the code optimized for the Zen4 core (requires AVX512F, AVX512VNNI, AVX512VL, AVX512BW, AVX512DQ).
> 
> All of those extensions are supported (and also AVX512_fp16 which AMD does not support even on Zen 5), none of the normal sources I use for this have been updated to show Granite Rapids but I did find [this](https://www.phoronix.com/image-viewer.php?id=intel-xeon-6980p-performance&image=intel_xeon_6980p_2_lrg). Granite rapids was supposed to have support for Intel AVX10 (Version 1, or Intel AVX10.1) but that apparently did not happen.
> 
> >I have seen a higher than usual amount of stars added to my repository in the last few days, I guess this must be due to your post.
> 
> I've also seen an uptick in organic mentions of ik_llama.cpp recently and have done my best to help people understand all the new features and benefits.
> 
> 👤 **ubergarm** replied the **2025-03-13** at **22:15:00**:<br>
> @ikawrakow 
> 
> > Very interesting results, thank you for posting and including my little LLM inference playground in the results.
> 
> My pleasure, thanks for sharing your work. I've been tracking progress across various inference engines and stumbled onto yours from [this github pr discussion](https://github.com/ggml-org/llama.cpp/pull/12227#issuecomment-2708219642) about MLA and flash attention.
> 
> > The tested 6980P has a theoretical bandwidth of 512? GiB/s
> 
> Your back of the napkin math is good, this machine tested with `mlc` (Intel Memory Latency Checker) shows just almost exactly 512GiB/s per CPU socket within the same NUMA node. Shown in the 1x NUMA node per CPU core here with BIOS set to `SNC=Disable`. Otherwise it has 3x nodes per CPU with an uneven number of cores hah...
> 
> ```
> Measuring Memory Bandwidths between nodes within system 
> Bandwidths are in MB/sec (1 MB/sec = 1,000,000 Bytes/sec)
> Using all the threads from each core if Hyper-threading is enabled
> Using Read-only traffic type
> 		Numa node
> Numa node	     0	     1	
>        0	554843.5	247793.1	
>        1	247281.1	552385.5	
> ```
> 
> > Playing with some of the more advanced options that mainline llama.cpp does not have would be of course very interesting too.
> 
> Yes, I'm playing with [ktransformers](https://github.com/ubergarm/r1-ktransformers-guide/) as well, but it has a hard requirement on GPU. Unfortunately, this 6980P rig has no GPU so I'm limited to CPU only testing.
> 
> > so one needs to go through the PRs to discover them
> 
> Correct, I have not gone through your branches and PRs to figure out the best combination of code and options for pure CPU inference using the various unsloth R1 671B GGUF quants.
> 
> @saood06 
> 
> > Also you can look at zts9989's comment https://github.com/ggml-org/llama.cpp/pull/11397#issuecomment-2716225570 where he talks about NUMA and what llama.cpp could improve on after he found that "approximately 50% of CPU usage is spent on thread synchronization" when running Deepseek R1 with multiple numa nodes.
> 
> Yes, this is the most optimized CPU implementation of which I've heard to date. Seems unlikely they will release code directly to github, but possibly would share files via email, but I haven't asked.
> 
> > All of those extensions are supported (and also AVX512_fp16 
> 
> Correct, I have the output of `lscpu` buried in the `Methodology and Notes` `<detail>` as you discovered. Copy pasted below for ease of reference. The three AMX Extensions specific flags unique to newer Intel Xeon are `amx_bf16` `amx_int8` `amx_tile`. Very interesting for DeepSeek is that Intel's next generation Diamond Rapids may support [`amx_fp8`](https://www.phoronix.com/news/Intel-AMX-FP8-In-LLVM). It's mildly annoying that older NVIDIA GPUs with capability <8.9 don't natively support fp8e4nv. This is required for [DeepSeek's Triton fp8_gemm implementation](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py). Then the official DeepGemm implementation seems limited to [only 9.0 (H100s) hardware](https://github.com/deepseek-ai/DeepGEMM/issues/6) currently too afaict.
> 
> Funny to see [guys with Dual 5090s whining](https://github.com/vllm-project/vllm/issues/14628#issuecomment-2720369467) that their stuff doesn't work yet haha....
> 
> It seems llama.cpp main has some support for these, however I'm not completely sure that it speeds up token generation or if it needs a specific quant. It does seem to at least be compiled in and doing *something* on the `Q8_0` test:
> 
> ```
> load_tensors: tensor 'token_embd.weight' (q8_0) (and 54 others) cannot be used with preferred buffer type AMX, using CPU instead
> ...
> load_tensors:          AMX model buffer size = 18214.39 MiB
> load_tensors:   CPU_Mapped model buffer size = 45565.90 MiB
> ...
> ```
> 
> I don't believe I noticed these debug logs when I tested `ik_llama.cpp@a48e1632` by simply compiling main branch with no special new arguments.
> 
> Quoting [@aubreyli](https://github.com/ggml-org/llama.cpp/discussions/12088#discussioncomment-12469251)
> > AMX tile config is [here](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/amx/mmq.cpp#L168) in llama.cpp And AMX MUL_MAT is [here](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/amx/mmq.cpp#L2369)
> > 
> > If the tensor OP type is GGML_OP_MUL_MAT, it will be invoked on Intel AMX supported platform.
> 
> I have more time soon with access to this dual 6980P if you have a specific branch, feature, or quant configuration suggestion for me to try or point me to a branch or PR and I can read-up on it to test and benchmark.
> 
> Thanks!
> 
> ```
> ## CPU
> $ lscpu | grep Xeon
> Model name:                           Intel(R) Xeon(R) 6980P
> 
> ## CPU Flags
> $ lscpu | grep Flags
> Flags: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cat_l2 cdp_l3 intel_ppin cdp_l2 ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local split_lock_detect user_shstk avx_vnni avx512_bf16 wbnoinvd dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req vnmi avx512vbmi umip pku ospke waitpkg avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg tme avx512_vpopcntdq la57 rdpid bus_lock_detect cldemote movdiri movdir64b enqcmd fsrm md_clear serialize tsxldtrk pconfig arch_lbr ibt amx_bf16 avx512_fp16 amx_tile amx_int8 flush_l1d arch_capabilities
> ```
> 
> 👤 **saood06** replied the **2025-03-13** at **22:51:58**:<br>
> > > Playing with some of the more advanced options that mainline llama.cpp does not have would be of course very interesting too.
> > 
> > Yes, I'm playing with [ktransformers](https://github.com/ubergarm/r1-ktransformers-guide/) as well, but it has a hard requirement on GPU. Unfortunately, this 6980P rig has no GPU so I'm limited to CPU only testing.
> 
> When you do have a machine with a GPU, ik_llama.cpp can also make use of it in a similar way by offloading select tensors to the GPU. The implementation here is a lot more flexible, but that comes at the cost of knowing what tensors to offload. I would be really interested to see how performance stacks up against ktransformers on the same machine, with both offloading to the GPU.
> 
> > Correct, I have not gone through your branches and PRs to figure out the best combination of code and options for pure CPU inference using the various unsloth R1 671B GGUF quants.
> 
> There is no best performance, MLA offers significantly better TG performance at long contexts but it does come at the cost of PP (as MLA is inherently more compute intensive) . There have been a lot of optimizations done by ikawrakow to help recover that PP performance, and I think the best for MLA currently is with the use of -mla 2 -fa. The -fmoe and -rtr flags also improve performance. (There might be a caveat with -rtr as it disables mmap and may do non optimal things with where memory is allocated, I personally repack my quants and do not use the -rtr flag)
> 
> >It's mildly annoying that older NVIDIA GPUs with capability <8.9 don't natively support fp8e4nv. This is required for [DeepSeek's Triton fp8_gemm implementation](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py). Then the official DeepGemm implementation seems limited to [only 9.0 (H100s) hardware](https://github.com/deepseek-ai/DeepGEMM/issues/6) currently too afaict.
> 
> I'm also annoyed by that as I have a 3090 and torch compile on fp8 stuff just errors instead of up casting.
> 
> 
> > It seems llama.cpp main has some support for these, however I'm not completely sure that it speeds up token generation or if it needs a specific quant. It does seem to at least be compiled in and doing _something_ on the `Q8_0` test:
> > 
> > ```
> > load_tensors: tensor 'token_embd.weight' (q8_0) (and 54 others) cannot be used with preferred buffer type AMX, using CPU instead
> > ...
> > load_tensors:          AMX model buffer size = 18214.39 MiB
> > load_tensors:   CPU_Mapped model buffer size = 45565.90 MiB
> > ...
> > ```
> > 
> > I don't believe I noticed these debug logs when I tested `ik_llama.cpp@a48e1632` by simply compiling main branch with no special new arguments.
> > 
> > Quoting [@aubreyli](https://github.com/ggml-org/llama.cpp/discussions/12088#discussioncomment-12469251)
> > 
> > > AMX tile config is [here](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/amx/mmq.cpp#L168) in llama.cpp And AMX MUL_MAT is [here](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/amx/mmq.cpp#L2369)
> > > If the tensor OP type is GGML_OP_MUL_MAT, it will be invoked on Intel AMX supported platform.
> > 
> 
> AMX support was added to llama.cpp after ik_llama.cpp last merged mainline. Some things are easy to port into ik_llama.cpp, others are more difficult, I have not looked into it but I also don't know how much value it would add given how ik_llama.cpp overhauls a lot of the backend anyways.
> 
> > I have more time soon with access to this dual 6980P if you have a specific branch, feature, or quant configuration suggestion for me to try or point me to a branch or PR and I can read-up on it to test and benchmark.
> 
> I'll leave requests to @ikawrakow but I think his table above showing off -fa -rtr, and -fmoe show the benefits of those arguments. This PR https://github.com/ikawrakow/ik_llama.cpp/pull/246 has a good summary of the MLA and FA options, and this latest PR shows the most recent numbers and latest optimization: https://github.com/ikawrakow/ik_llama.cpp/pull/253

---

👤 **saood06** replied the **2025-03-25** at **03:29:01**:<br>

@ubergarm (thought you might also be interested in this).

>[KTransformers](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md#some-explanations) just duplicates matrices on each NUMA domain !

Someone has shared code that can duplicate the model for NUMA benefits on llama.cpp:

https://github.com/ggml-org/llama.cpp/discussions/12289

>TLDR: Replicate models on each NUMA. On my platform, pure CPU inference of QwQ-32B FP16 improved from ~6.6 token/s to ~10.7 token/s, and DeepSeek R1 671B Q8 from ~7.2 token/s to ~9.7 token/s. You can find the modified llama.cpp version [here](https://github.com/vproxy-tools/llama.cpp).

The downside of duplicating the model is pretty heavy, but this approach obviously avoids any non local memory access, and shows the upper bound on performance that that could be gained from other solutions that reduce or remove non local memory access.

Looking at the codebase, I think it currently only works for dual socket nodes, and I would have been more interested in testing it but none of my machines (even the very unstable one quad socket 1 TB memory node that I haven't turned on in a long time) would have enough RAM to replicate my preferred quant of R1, I'd have to use one under 192 GB (I do still have my IQ1_S_R4 V2 that is 129 GB).

> 👤 **ubergarm** replied the **2025-03-25** at **15:58:04**:<br>
> Super, I just fetched this fork and will take a peek.
> 
> > The downside of duplicating the model is pretty heavy
> 
> Yeah, it is *so much* RAM! 
> 
> Probably easiest to go BIOS `NPS1` on dual socket AMD Epyc or on newer Intel Xeon BIOS `SNC=Disable` to get exactly 2 big NUMA nodes (one per CPU socket). Ideally you would have the most number of individual NUMA nodes to maximize performance, but the RAM is then too small per node to fit the bigger models.
> 
> Also [mingfeima](https://github.com/mingfeima) left an [interesting comment](https://github.com/ggml-org/llama.cpp/issues/12003#issuecomment-2731572966) recently discussing some of the intel specific optimizations and work he's doing on sglang.
> 
> Finally, I recently saw Wendell of [level1techs youtube channel do a video](https://www.youtube.com/watch?v=kOh04PhXqmY) about quad socket Intel Xeon. Seems like it could be configured into 8 individual NUMA nodes with 1TB each possibly?  Talk about wasting RAM, but would be fun to try haha...
> 
> 👤 **saood06** replied the **2025-03-27** at **07:24:15**:<br>
> >Super, I just fetched this fork and will take a peek.
> 
> Did you ever test it?

---

👤 **ikawrakow** replied the **2025-03-25** at **16:06:42**:<br>

>  Ideally you would have the most number of individual NUMA nodes to maximize performance,

Why?

> 👤 **ubergarm** replied the **2025-03-25** at **16:14:54**:<br>
> Looking at Intel Memory Latency Checker `mlc` benchmarks suggest that the memory local to the compute on a specific NUMA node gives best bandwidth and latency.
> 
> My thinking is that duplicating weights into each NUMA node and having local threads working with that RAM would maximize performance.
> 
> However, I'm not fully aware of the other implications of combining computations for the final results in this "data parallel" situation. I've only read about "all reduce" in GPU specific implementations suggesting `nvlink` or `p2p` or RDMA infiniband networking is required for those "tensor parallel" implementations.
> 
> For now I'd be happy to configure each CPU socket as a single numa node in BIOS as that would probably be good enough and more likely to have enough RAM to fit bigger models. So data parallel = number CPU sockets = (probably 2 for most folks)

---

👤 **ikawrakow** replied the **2025-03-25** at **16:24:17**:<br>

Sure, that would be if you wanted to squeeze out the last bit of performance. But we are not at that stage. Instead, we are a factor of 2 or more away from what should be possible. Having 2 big NUMA nodes would make the distribution of weights much easier: simply change the weight loading to use two threads, each pinned to a specific NUMA node, and each loading half of the tensor data. During inference pin half the threads to run on the 1st NUMA node, and the other half to the second NUMA node. My thinking is that this should give a significant boost in performance without replicating the model on both NUMA nodes. It is of course possible to do stuff such as this with several NUMA nodes, but it makes things way more complicated. So, I'm thinking that the 1st step should be to get better performance with 2 NUMA nodes. But if you are telling me that this is very far from ideal, and that the only way to get better performance is to enable and utilize all NUMA nodes, then it is a waste of time to implement the simple approach described above.

> 👤 **ubergarm** replied the **2025-03-25** at **16:36:46**:<br>
> > that would be if you wanted to squeeze out the last bit of performance. But we are not at that stage.
> 
> Yes, I agree on both points.
> 
> >  I'm thinking that the 1st step should be to get better performance with 2 NUMA nodes
> 
> Again, I agree. My understanding is ktransformers `USE_NUMA=1` compilation flag is for 2 NUMA nodes. Also the [discussion/fork saood06 linked](https://github.com/ggml-org/llama.cpp/discussions/12289) seems to be specific to 2 NUMA nodes.
> 
> Going for exactly 2 NUMA nodes is also good because:
> 1. Most AMD Epyc BIOS dual socket boards likely support `NPS1` for exactly 2 NUMA Nodes
> 2. Newer Intel Xeon BIOS dual socket boards supports `SNC=Disable`for exactly 2 NUMA Nodes
> 
> No need to worry about rare brand new quad socket intel xeon boards or more smaller NUMA nodes currently imo.
> 
> I'll try to find my `mlc` benchmarks and post here, as the bandwidth is still pretty good converting a single CPU into 1 NUMA node.
> 
> 👤 **ubergarm** replied the **2025-03-25** at **16:52:11**:<br>
> #### intel `mlc`
> 
> Configuring BIOS to `SNC=Disable` to collapse 3x NUMA nodes per CPU socket into a single NUMA node per 6980P socket gives similar enough RAM bandwidth/latency performance.
> 
> So probably not worth trying to support more than 2 NUMA nodes "data parallel" type feature assuming other systems perform similarly.
> 
> <details>
> 
> <summary>Dual Socket Intel Xeon 6980P `SNC=Auto/Enabled`</summary>
> This gives 6x total NUMA nodes (3x per CPU socket).
> 
> ```
> Intel(R) Memory Latency Checker - v3.11b
> Measuring idle latencies for sequential access (in ns)...
> 		Numa node
> Numa node	     0	     1	     2	     3	     4	     5	
>        0	 138.7	 168.0	 208.5	 394.1	 475.2	 445.1	
>        1	 160.3	 134.4	 170.4	 415.2	 448.2	 479.7	
>        2	 156.2	 123.6	 106.5	 507.8	 513.2	 452.5	
>        3	 396.0	 476.0	 445.6	 102.0	 129.4	 157.5	
>        4	 419.7	 452.6	 421.2	 122.1	 102.4	 130.2	
>        5	 445.4	 449.5	 392.4	 148.3	 122.3	 103.8	
> 
> Measuring Peak Injection Memory Bandwidths for the system
> Bandwidths are in MB/sec (1 MB/sec = 1,000,000 Bytes/sec)
> Using all the threads from each core if Hyper-threading is enabled
> Using traffic with the following read-write ratios
> ALL Reads        :	1126026.6	
> 3:1 Reads-Writes :	972377.5	
> 2:1 Reads-Writes :	933247.3	
> 1:1 Reads-Writes :	927164.2	
> Stream-triad like:	939630.2	
> 
> Measuring Memory Bandwidths between nodes within system 
> Bandwidths are in MB/sec (1 MB/sec = 1,000,000 Bytes/sec)
> Using all the threads from each core if Hyper-threading is enabled
> Using Read-only traffic type
> 		Numa node
> Numa node	     0	     1	     2	     3	     4	     5	
>        0	187911.4	188622.8	188716.9	94137.8	93596.5	93730.5	
>        1	188260.8	188176.4	188653.1	94495.4	90659.3	93774.2	
>        2	188624.6	188626.7	188129.6	94509.6	27886.4	93792.7	
>        3	94161.1	93415.7	94558.3	187851.4	188418.6	188691.9	
>        4	94201.1	91712.7	94546.8	188169.2	188067.6	188544.2	
>        5	94183.2	44861.0	94241.8	188416.4	188380.0	187933.8	
> 
> Measuring Loaded Latencies for the system
> Using all the threads from each core if Hyper-threading is enabled
> Using Read-only traffic type
> Inject	Latency	Bandwidth
> Delay	(ns)	MB/sec
> ==========================
>  00000	378.26	1125007.8
>  00002	381.36	1125706.3
>  00008	382.90	1125594.5
>  00015	381.40	1128101.6
>  00050	377.79	1129501.1
>  00100	296.51	1117783.2
>  00200	301.72	1122699.0
>  00300	207.87	1017250.0
>  00400	170.76	 782113.4
>  00500	157.40	 665276.4
>  00700	138.25	 488635.4
>  01000	128.65	 349546.6
>  01300	125.55	 271876.5
>  01700	123.93	 209644.5
>  02500	116.19	 143990.9
>  03500	120.17	 103477.5
>  05000	119.53	  72875.8
>  09000	113.89	  40898.3
>  20000	115.14	  18113.6
> 
> Measuring cache-to-cache transfer latency (in ns)...
> Local Socket L2->L2 HIT  latency	80.5
> Local Socket L2->L2 HITM latency	80.9
> Remote Socket L2->L2 HITM latency (data address homed in writer socket)
> 			Reader Numa Node
> Writer Numa Node     0	     1	     2	     3	     4	     5	
>             0	     -	  99.3	 124.9	 376.2	 401.7	 429.5	
>             1	 108.8	     -	 100.9	 452.1	 425.7	 422.2	
>             2	 131.0	 103.8	     -	 435.5	 407.4	 378.1	
>             3	 372.3	 393.3	 423.4	     -	 101.2	 125.6	
>             4	 444.2	 414.2	 413.5	 106.3	     -	 100.9	
>             5	 429.5	 399.3	 374.0	 130.3	 106.1	     -	
> Remote Socket L2->L2 HITM latency (data address homed in reader socket)
> 			Reader Numa Node
> Writer Numa Node     0	     1	     2	     3	     4	     5	
>             0	     -	 109.6	 140.2	 381.2	 444.0	 440.0	
>             1	 106.9	     -	 110.8	 405.8	 414.7	 411.6	
>             2	 137.1	 103.8	     -	 436.3	 442.6	 381.2	
>             3	 380.8	 441.6	 439.1	     -	 110.6	 139.5	
>             4	 406.3	 412.7	 411.6	 105.8	     -	 110.7	
>             5	 436.7	 440.5	 381.2	 136.3	 105.9	     -	
> 
> ```
> 
> </details>
> 
> ---
> 
> <details>
> 
> <summary>Dual Socket Intel Xeon 6980P `SNC=Disabled`</summary>
> 
> This gives 2x total NUMA nodes (1x per CPU socket).
> 
> ```
> Intel(R) Memory Latency Checker - v3.11b
> Measuring idle latencies for sequential access (in ns)...
> 		Numa node
> Numa node	     0	     1	
>        0	 130.7	 449.2	
>        1	 410.0	 129.4	
> 
> Measuring Peak Injection Memory Bandwidths for the system
> Bandwidths are in MB/sec (1 MB/sec = 1,000,000 Bytes/sec)
> Using all the threads from each core if Hyper-threading is enabled
> Using traffic with the following read-write ratios
> ALL Reads        :	1108235.0	
> 3:1 Reads-Writes :	972151.5	
> 2:1 Reads-Writes :	940099.8	
> 1:1 Reads-Writes :	928269.2	
> Stream-triad like:	918997.2	
> 
> Measuring Memory Bandwidths between nodes within system 
> Bandwidths are in MB/sec (1 MB/sec = 1,000,000 Bytes/sec)
> Using all the threads from each core if Hyper-threading is enabled
> Using Read-only traffic type
> 		Numa node
> Numa node	     0	     1	
>        0	554843.5	247793.1	
>        1	247281.1	552385.5	
> 
> Measuring Loaded Latencies for the system
> Using all the threads from each core if Hyper-threading is enabled
> Using Read-only traffic type
> Inject	Latency	Bandwidth
> Delay	(ns)	MB/sec
> ==========================
>  00000	357.28	1106966.8
>  00002	362.94	1108392.3
>  00008	363.07	1107547.6
>  00015	360.97	1104844.6
>  00050	359.09	1102679.2
>  00100	307.11	1099803.6
>  00200	320.42	1105411.1
>  00300	231.07	1007100.3
>  00400	188.93	 789261.0
>  00500	174.05	 665122.5
>  00700	158.95	 487463.0
>  01000	150.90	 349530.7
>  01300	148.47	 271576.2
>  01700	146.67	 209392.6
>  02500	144.40	 143857.9
>  03500	142.66	 103386.9
>  05000	140.57	  72810.8
>  09000	139.24	  40768.0
>  20000	138.79	  18002.4
> 
> Measuring cache-to-cache transfer latency (in ns)...
> Local Socket L2->L2 HIT  latency	179.7
> Local Socket L2->L2 HITM latency	180.2
> Remote Socket L2->L2 HITM latency (data address homed in writer socket)
> 			Reader Numa Node
> Writer Numa Node     0	     1	
>             0	     -	 433.3	
>             1	 413.7	     -	
> Remote Socket L2->L2 HITM latency (data address homed in reader socket)
> 			Reader Numa Node
> Writer Numa Node     0	     1	
>             0	     -	 425.0	
>             1	 422.4	     -	
> ```
> 
> </details>
> 
> ## References
> * [Additional Benchmarks and discussions on Phoronix](https://www.phoronix.com/review/xeon-6980p-snc3-hex)
> 
> 👤 **saood06** replied the **2025-03-25** at **18:09:30**:<br>
> > During inference pin half the threads to run on the 1st NUMA node, and the other half to the second NUMA node.
> 
> The problem is not splitting the model, it is ensuring the work of any given thread is stored local to it's NUMA node. 
> 
> This PR: https://github.com/ggml-org/llama.cpp/pull/6915 made it difficult as mentioned here: https://github.com/ggml-org/llama.cpp/issues/1437#issuecomment-2095809308
> 
> Maybe you could use [this](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-0/thread-affinity-interface.html#LOW_LEVEL_AFFINITY_API) so that each thread could change it's affinity to a random thread on the correct numa node (this would also work since I don't think this would otherwise be compatible with --numa interleave [but not sure has been a long time since I looked into that).
> 
> 👤 **ikawrakow** replied the **2025-03-25** at **18:17:01**:<br>
> There is no dynamic thread scheduling here. No thread pools either.
> 
> In my experience from the past, touching memory with on a NUMA node makes it automatically that the actual data is stored in a memory bank local to the node on which the thread is running.  The difficulty will be more in fighting with the almighty `ggml` backend than anything else.
> 
> 👤 **ikawrakow** replied the **2025-03-25** at **18:26:08**:<br>
> Dynamic thread scheduling does help for PP with big enough batch sizes. It would also help on systems with a mix of P/E cores (although, if mainline `llama.cpp` has that, I notice absolutely zero benefit on my M2-Max. Performance there is still best with 8 threads, not 12). But for TG with all same cores the overhead of thread synchronization for work stealing is typically too high to have benefit. Maybe it is different for a humongous model such as DeepSeek-R1? But then again, it has nearly 4X the number of nodes in the compute graph, so the work per node is not that much higher than DeepSeek-Lite.
> 
> 👤 **saood06** replied the **2025-03-25** at **18:36:09**:<br>
> > There is no dynamic thread scheduling here. No thread pools either.
> 
> @bmtwl
> 
> You said 
> 
> >The problem at that time was the thread allocation code didn't have any way to ascertain which numa node it was running on or what numa node the tensors it was going to be working on was pinned to.
> >[...]
> >I'm still very interested in this and want to take another stab at it, but haven't been able to work up the will to try again yet.
> 
> Do you think you'd want to attempt it in this repo as there is no dynamic scheduling or threadpool here?

---

👤 **ubergarm** replied the **2025-03-30** at **17:25:05**:<br>

Oh I see a benchmark in the wild attempting to benchmark that [vproxy-tools/llama.cpp](https://github.com/vproxy-tools/llama.cpp) NUMA data parallel code against ik fork: https://github.com/ggml-org/llama.cpp/discussions/12289#discussioncomment-12668490

> It seems clear that porting the mirror impl. to the ik fork should make the best available version.

Not sure the details of how they are running it though...

> 👤 **saood06** replied the **2025-03-30** at **20:58:05**:<br>
> > Oh I see a benchmark in the wild attempting to benchmark that [vproxy-tools/llama.cpp](https://github.com/vproxy-tools/llama.cpp) NUMA data parallel code against ik fork: [ggml-org/llama.cpp#12289 (comment)](https://github.com/ggml-org/llama.cpp/discussions/12289#discussioncomment-12668490)
> > 
> > Not sure the details of how they are running it though...
> 
> Thanks for the link, I agree it would be nice if they included more details.
> 
> 👤 **ubergarm** replied the **2025-03-30** at **21:14:31**:<br>
> Yeah, I gave it a try and while it did run it wasn't allocating threads on both NUMA nodes so I gave up for now after posting my logs.
> 
> 👤 **saood06** replied the **2025-03-30** at **21:34:22**:<br>
> > Yeah, I gave it a try and while it did run it wasn't allocating threads on both NUMA nodes so I gave up for now after posting my logs.
> 
> Did you try running it with numactl on just 2 NUMA nodes? There is also an issue tracker for [vproxy-tools/llama.cpp](https://github.com/vproxy-tools/llama.cpp/issues) where you could report that.

---

👤 **bhugueney** replied the **2025-04-08** at **10:24:55**:<br>

I currently settle for running my DeepSeek v3 model on just one NUMA / socket of my dual socket system. However, while investigating the draft models situation, it occurred to me that if should be relatively easy to specify cores for the main model (on one socket) and specify other cores (in my case on the other socket/NUMA node) for the draft model as communication between the two should be minimal.
What do people think about it?

---

👤 **saood06** replied the **2025-05-20** at **08:37:01**:<br>

On my dual socket machine using https://github.com/intel/pcm

I found this is what it looks like during PP:

|  | READ (GB)  | WRITE (GB) | LOCAL | CPU energy | DIMM energy | LLCRDMISSLAT (ns) | UncFREQ (Ghz) |
|------------|-------|-------|-------|------------|-------------|-------------------|---------------|
| Socket - 0     | 7.93  | 3.60  | 49 %  | 96.90      | 23.78       | 365.82            | 2.30          |
| Socket - 1      | 2.56  | 1.55  | 46 %  | 89.43      | 18.93       | 436.65            | 2.21          |
| Total      | 10.50 | 5.15  | 48 %  | 186.32     | 42.71       | 400.13            | 2.25          |

And during TG:

|  | READ (GB)  | WRITE (GB) | LOCAL | CPU energy | DIMM energy | LLCRDMISSLAT (ns) | UncFREQ (Ghz) |
|------------|-------|-------|-------|------------|-------------|-------------------|---------------|
| Socket - 0   | 16.22 | 0.55  | 90 %  | 134.39     | 26.05       | 219.40            | 2.68          |
| Socket - 1       | 14.74 | 0.15  | 95 %  | 133.64     | 25.46       | 214.65            | 2.77          |
| Total      | 30.96 | 0.70  | 92 %  | 268.02     | 51.52       | 216.97            | 2.73          |

---

👤 **VinnyG9** replied the **2025-05-21** at **04:15:29**:<br>

just sharing i tried all snoop modes on my x99 dual board and got 200-300% boost vs stock bios settings, this setting is also available on xeon scalable fwiw

## stock bios
| model                             |      size |  params | backend | ngl | threads | fa | rtr | fmoe |   test |            t/s |
| ----------------------------------- | ----------: | --------: | --------- | ----: | --------: | ---: | ----: | -----: | -------: | ---------------: |
| ============ Repacked 337 tensors |           |         |         |     |         |    |     |      |        |                |
| qwen3moe ?B Q4_K - Medium         | 16.49 GiB | 30.53 B | CUDA    |   0 |      31 |  1 |   1 |    1 |  pp256 | 108.42 ± 1.82 |
| qwen3moe ?B Q4_K - Medium         | 16.49 GiB | 30.53 B | CUDA    |   0 |      31 |  1 |   1 |    1 |  pp512 | 123.10 ± 1.64 |
| qwen3moe ?B Q4_K - Medium         | 16.49 GiB | 30.53 B | CUDA    |   0 |      31 |  1 |   1 |    1 | pp1024 | 118.61 ± 1.67 |
| qwen3moe ?B Q4_K - Medium         | 16.49 GiB | 30.53 B | CUDA    |   0 |      31 |  1 |   1 |    1 |  tg128 |  12.28 ± 0.03 |
| qwen3moe ?B Q4_K - Medium         | 16.49 GiB | 30.53 B | CUDA    |   0 |      31 |  1 |   1 |    1 |  tg256 |  12.17 ± 0.06 |

## home snoop w/ dir OSB

| model                             |      size |  params | backend | ngl | threads | fa | rtr | fmoe |  test |             t/s |
| ----------------------------------- | ----------: | --------: | --------- | ----: | --------: | ---: | ----: | -----: | ------: | ----------------: |
| ============ Repacked 337 tensors |           |         |         |     |         |    |     |      |       |                 |
| qwen3moe ?B Q4_K - Medium         | 16.49 GiB | 30.53 B | CUDA    |   0 |      31 |  1 |   1 |    1 |  pp64 | 173.70 ± 16.62 |
| qwen3moe ?B Q4_K - Medium         | 16.49 GiB | 30.53 B | CUDA    |   0 |      31 |  1 |   1 |    1 | pp128 | 235.53 ± 19.14 |
| qwen3moe ?B Q4_K - Medium         | 16.49 GiB | 30.53 B | CUDA    |   0 |      31 |  1 |   1 |    1 | pp256 |  270.99 ± 7.79 |
| qwen3moe ?B Q4_K - Medium         | 16.49 GiB | 30.53 B | CUDA    |   0 |      31 |  1 |   1 |    1 | pp512 |  263.82 ± 6.02 |
| qwen3moe ?B Q4_K - Medium         | 16.49 GiB | 30.53 B | CUDA    |   0 |      31 |  1 |   1 |    1 |  tg64 |   31.61 ± 1.01 |
| qwen3moe ?B Q4_K - Medium         | 16.49 GiB | 30.53 B | CUDA    |   0 |      31 |  1 |   1 |    1 | tg128 |   34.76 ± 1.54 |
| qwen3moe ?B Q4_K - Medium         | 16.49 GiB | 30.53 B | CUDA    |   0 |      31 |  1 |   1 |    1 | tg256 |   35.70 ± 0.34 |

> 👤 **ubergarm** replied the **2025-05-21** at **14:26:30**:<br>
> Wow, big gains! I'd never heard of "snoop" mode, but don't have a lot of intel server experience:
> 
> > DIR+OSB mode allows for low local memory latency, high local memory bandwidth and I/O directory cache to reduce directory update overheads for I/O accesses.
> 
> Are you running hybrid CPU+GPU CUDA offloading some layers? I forget your exact system specs and VRAM, but if you can offload the whole thing it can go quite faster psure.  Also, if I'm running CPU/RAM *only* I generally recompile and disable CUDA backend fwiw.
> 
> Glad you're having fun tweaking and tuning!
> 
> 👤 **VinnyG9** replied the **2025-05-21** at **18:07:27**:<br>
> > Wow, big gains! I'd never heard of "snoop" mode, but don't have a lot of intel server experience:
> > 
> > > DIR+OSB mode allows for low local memory latency, high local memory bandwidth and I/O directory cache to reduce directory update overheads for I/O accesses.
> > 
> > Are you running hybrid CPU+GPU CUDA offloading some layers? I forget your exact system specs and VRAM, but if you can offload the whole thing it can go quite faster psure. Also, if I'm running CPU/RAM _only_ I generally recompile and disable CUDA backend fwiw.
> > 
> > Glad you're having fun tweaking and tuning!
> 
> i saw ik recommending it so i tried disabling cuda build for cpu inference, but up to 2k tokens max i tested it was slower no idea why
>  snoop mode is a numa thing but it helped on single cpu inference also by ~10-30% , i see a nice boost on intel MLC too like 116 > 140GB/s
> hybrid inference only saw ~10% TG increase(offloading about 40% of weghts)
> 
> qwen3 dense got a 90% boost