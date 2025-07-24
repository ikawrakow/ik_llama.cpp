### 🔀 [#441](https://github.com/ikawrakow/ik_llama.cpp/pull/441) - Trellis quants with CPU inference

| **Author** | `andrewkchan` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-20 |
| **Updated** | 2025-05-23 |

---

#### Description

As requested a while ago, takes (https://github.com/ikawrakow/ik_llama.cpp/pull/113) and adds CPU implementations of the quantized matmuls (via iqk_mul_mat) for inference. AVX2 and F16C support are required.

As predicted, the CPU ops are very slow. For Llama-3.1-8B-Instruct, I get ~0.3~ 4.83 t/s with IQ2_KT compared to ~>1.0~ 4.59 t/s with F16 on AMD EPYC 7R32 (32 cores). Note I am not a SIMD expert and have only spent moderate time on optimizations (e.g. basic use of AVX2/F16C, flattening of the trellis iterations), so it may be possible to speed things up. I also have not added implementations for `HAVE_FANCY_SIMD`. Additionally, there are only mulmats for F32 activations, as that is what the 3INST algorithm returns (as pointed out in the original PR description).

I am not sure of the PR practices - if you'd like me to merge into https://github.com/ikawrakow/ik_llama.cpp/pull/113 rather than the main branch, happy to change. I also tried to clean up some of the comments / dead code in the WIP branch, but can revert those changes as well.

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [ ] Low
  - [X] Medium
  - [ ] High

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-21** at **07:13:48**:<br>

> For Llama-3.1-8B-Instruct, I get 0.3t/s with IQ2_KT compared to >1.0t/s with F16 on AMD EPYC 7R32 (32 cores)

Is this in debug mode? I'm getting 10.4 t/s for `IQ2_KT` on my 16-core Ryzen-7950X CPU. Which (as expected) is slow for a 2-bit quantized 8B model, but still in the acceptable range.

---

👤 **andrewkchan** commented the **2025-05-21** at **07:17:47**:<br>

I'm compiling with `cmake --build ./build --config Release -j $(nproc)`. I might need to tweak the number of threads; I've found this greatly impacts performance on my test machine in the past for llama.cpp.

Here's how I'm testing:
```
alias ik-build='cmake --build ./build --config Release -j $(nproc)'
ik-build && ./build/bin/llama-cli -m ../Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct-IQ2_KT-2.gguf -cnv -p "You are a helpful assistant" -ngl 0 -c 4096

<prompt with something like "1+1=" then CTRL+C after several tokens are generated to get the numbers>
```

Should I be using llama-bench or some other tool?

---

👤 **ikawrakow** commented the **2025-05-21** at **07:24:07**:<br>

I also tried `llama-cli` to make sure the output is coherent, and also get in the range of 10 t/s. To measure performance I now tend to use `llama-sweep-bench`. For instance, the table below was generated using
```
./bin/llama-sweep-bench -m iq2kt.bin -c 2560 -t 16 -fa -ctk q8_0 -ctv q8_0
```

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   11.436 |    44.77 |   12.278 |    10.42 |
|   512 |    128 |    512 |   10.743 |    47.66 |   12.782 |    10.01 |
|   512 |    128 |   1024 |   10.639 |    48.13 |   13.189 |     9.70 |
|   512 |    128 |   1536 |   11.668 |    43.88 |   13.185 |     9.71 |
|   512 |    128 |   2048 |   10.462 |    48.94 |   13.310 |     9.62 |

We get PP and TG performance as a function of the number of tokens in the KV cache `N_KV`.

---

👤 **andrewkchan** commented the **2025-05-21** at **07:30:16**:<br>

Ok, well it's great to know the CPU inference performance is not totally unusable and that it's probably just my setup! I will try to figure this out on my own. Might email you some more questions to not pollute this PR discussion. Thanks also for the pointer on benchmarking.

---

👤 **andrewkchan** commented the **2025-05-21** at **08:11:09**:<br>

I purged my build directory + recompiled and performance is a lot better, and I no longer see the weird `ggml_backend_sched_alloc_splits: failed to allocate graph` messages from (https://github.com/ggml-org/llama.cpp/discussions/8088). Possibly the build cache was using some artifacts from a previous debug build. 

Now F16 gets almost 4x faster at 4.59 generation t/s, and IQ2_KT now beats F16 at 4.83 generation t/s for me.

---

👤 **ikawrakow** commented the **2025-05-21** at **14:35:39**:<br>

I did speed up `IQ2_KT` slightly, see [this branch](https://github.com/ikawrakow/ik_llama.cpp/tree/ik/andrew_trellis). Here is what I get now on the Ryzen-7950X

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    8.176 |    62.62 |   10.268 |    12.47 |
|   512 |    128 |    512 |    8.312 |    61.60 |   10.476 |    12.22 |
|   512 |    128 |   1024 |    8.826 |    58.01 |   10.625 |    12.05 |
|   512 |    128 |   1536 |    8.453 |    60.57 |   10.704 |    11.96 |
|   512 |    128 |   2048 |    8.488 |    60.32 |   10.798 |    11.85 |

Overall it looks good to me, so we can think about merging. But there is also PR #435, where I have completely refactored `iqk_mul_mat.cpp`. Do you want to look into adding the changes on that branch?

---

👤 **andrewkchan** commented the **2025-05-22** at **04:32:39**:<br>

Terrific, this gets my test machine to 5.59t/s. I saw the LCG ops in next8 taking up lots of time but wasn't sure what to do about it, this is a cool trick - I assume having the constants as locals keeps them in registers or otherwise ensures they remain hot in cache?

Re: https://github.com/ikawrakow/ik_llama.cpp/pull/435 - it looks not too difficult to me to reconcile my new kernels with the refactor. If you're done with your refactor already, you could merge your PR and then I can fix the conflicts accordingly - maybe that's the cleanest way to do this?

---

👤 **ikawrakow** submitted a review the **2025-05-23** at **06:17:15**: ✅ `APPROVED`