### 🔀 [#241](https://github.com/ikawrakow/ik_llama.cpp/pull/241) - DeepSeek CUDA Flash Attention 

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-04 |
| **Updated** | 2025-03-06 |

---

#### Description

This PR makes the CUDA FA implementation work when the V head size is not the same as the K head size (e.g., DeepSeek-Lite/V3/R1).

For TG I had to set the FA precision to `F32`, else we get gibberish. Not sure if it is really a matter of insufficient precision, or if I have missed something in the `f16` vector kernel.

The PR implements FA just for standard attention. FA for MLA is left for a follow up PR.

Here the mandatory performance comparisons. Model is `IQ4_NL` quantized DeepSeek-Lite, GPU is RTX-4080.

First prompt processing as a function of prompt length. It is a MoE model where it is better to use larger `u_batch` sizes, so all calculations are for `u_batch = 2048`, except no-FA for `pp16384` where I had to use `u_batch = 1024` to not run out of GPU memory.

 | model                | fmoe |      test |    t/s (no FA)   |     t/s (FA)     |  Speedup |
| ---------------------| ---: | --------: | ---------------: | ---------------: | -------: |
| deepseek2 16B IQ4_NL |    1 |     pp512 |  4106.17 ± 78.36 |  4180.10 ± 79.78 |  1.018   |
| deepseek2 16B IQ4_NL |    1 |    pp1024 | 5473.08 ± 100.23 |  5875.54 ± 79.86 |  1.074   |
| deepseek2 16B IQ4_NL |    1 |    pp2048 |  5943.17 ± 43.21 | 7200.13 ± 105.52 |  1.211   |
| deepseek2 16B IQ4_NL |    1 |    pp4096 |  5229.14 ± 81.15 |  6750.99 ± 48.49 |  1.291   |
| deepseek2 16B IQ4_NL |    1 |    pp8192 |  4275.60 ± 45.58 |  6277.33 ± 26.04 |  1.468   |
| deepseek2 16B IQ4_NL |    1 |   pp16384 |  2970.70 ± 31.45 |  5479.87 ± 49.10 |  1.845   |

Nice gains increasing with prompt length.

Here is TG performance for 128 tokens as a function of tokens in the KV cache (preceding prompt length):

| model                | fmoe |          test |      t/s (no FA) |     t/s (FA)     |  Speedup |
| ---------------------| ---: | ------------: | ---------------: | ---------------: | -------: |
| deepseek2 16B IQ4_NL |    1 |   tg128@pp128 |    131.11 ± 0.06 |    135.26 ± 0.02 |  1.032   |   
| deepseek2 16B IQ4_NL |    1 |   tg128@pp256 |    130.10 ± 0.07 |    133.89 ± 0.37 |  1.029   |   
| deepseek2 16B IQ4_NL |    1 |   tg128@pp512 |    127.42 ± 0.05 |    132.17 ± 0.06 |  1.037   |   
| deepseek2 16B IQ4_NL |    1 |  tg128@pp1024 |    121.39 ± 0.22 |    127.59 ± 0.06 |  1.051   |   
| deepseek2 16B IQ4_NL |    1 |  tg128@pp2048 |    116.00 ± 0.32 |    119.93 ± 0.19 |  1.034   |   
| deepseek2 16B IQ4_NL |    1 |  tg128@pp4096 |    106.77 ± 0.47 |    107.60 ± 0.10 |  1.008   |   
| deepseek2 16B IQ4_NL |    1 |  tg128@pp8192 |     89.56 ± 0.20 |     89.57 ± 0.22 |  1.000   |
| deepseek2 16B IQ4_NL |    1 | tg128@pp16384 |     66.23 ± 0.06 |     68.12 ± 0.24 |  1.028   |

Here the gains are very modest and, somewhat surprisingly, do not increase with KV cache size. I suspect the kernel is FA TG kernel is sub-optimal. It was inherited from mainline `llama.cpp` and all I did is adjust the kernel template parameter `D` (head size) to be either `Dk` (K head size) or `Dv` (V head size) depending on context. A better kernel for `Dk != Dv` is left for another day. For now we enjoy the benefit of much reduced compute buffer size. 

To limit the already excessive CUDA build time, I have only allowed K- and V-cache both `fp16` or `Q8_0`.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-03-04** at **09:51:25**:<br>

I'm by no means a CUDA programming expert, so I thought it is interesting to see if a CUDA beginner can compete with `llama.cpp` CUDA performance where there is an actual CUDA expert making continuous improvements. Here is a comparison between this PR and mainline `llama.cpp`  (latest build as of this writing, `build: 1a24c462 (4820)`). Mainline `llama-bench` does not have the `-gp` option to measure TG performance for a given KV cache size, so to simulate the presence of some not negligible KV cache, I use `tg1024` for TG performance. 
| model                |       test |     t/s (llama.cpp)  |  t/s (ik_llama)  |  Speedup  |
| ---------------------| ---------: | -------------------: | ---------------: | --------: |
| deepseek2 16B IQ4_NL |      pp512 |      3321.87 ± 32.74 |  4535.10 ± 79.21 |  1.365    |
| deepseek2 16B IQ4_NL |     pp1024 |     4191.67 ± 105.23 |  6189.62 ± 43.02 |  1.477    |
| deepseek2 16B IQ4_NL |     pp2048 |      4664.54 ± 84.49 |  7603.00 ± 26.43 |  1.630    |
| deepseek2 16B IQ4_NL |     pp4096 |      4203.41 ± 70.68 |  7300.89 ± 12.54 |  1.737    |
| deepseek2 16B IQ4_NL |     pp8192 |       3656.88 ± 3.05 |  6720.55 ± 12.22 |  1.838    |
| deepseek2 16B IQ4_NL |    pp16384 |      2642.45 ± 25.79 |  5796.02 ± 25.57 |  2.193    |
| deepseek2 16B IQ4_NL |     tg1024 |       132.66 ± 0.31  |   150.03 ± 0.02  |  1.131    |

For `pp512`, where FA has a (nearly) negligible impact on performance, the 36% gain comes from `-fmoe` (fused MoE `ffn_up, ffn_gate, ffn_down` operation). For long prompts FA is the main contributor (but `fmoe` still contributes in non-negligible ways). Interesting to note that there is a 13% performance benefit for TG despite the fact that TG is mostly memory bound (especially on the RTX-4080, which has a lot of computing power but just 768 GB/s of memory bandwidth). 

Why are the `ik_llama.cpp` values different from the above tables? For the PR text I did the performance comparisons on a computer with CUDA toolkit 12.4. Building latest `llama.cpp` with that failed, so I went to another machine with the same GPU but faster CPU. Also, to make sure that mainline can run the DeepSeek model, I quantized with `llama.cpp`, and this produces a different quantization (no extra bits spent on the attention tensors, which leads to higher performance).

---

👤 **davidsyoung** commented the **2025-03-04** at **19:08:54**:<br>

Cooking! Serious good work. I don't believe there's any package that has FA implemented like this yet.

---

👤 **davidsyoung** commented the **2025-03-06** at **15:02:29**:<br>

This PR from mainline llama.cpp may help with implementing MLA FA https://github.com/ggml-org/llama.cpp/pull/12227

---

👤 **ikawrakow** commented the **2025-03-06** at **15:24:03**:<br>

> This PR from mainline llama.cpp may help with implementing MLA FA https://github.com/ggml-org/llama.cpp/pull/12227

Ha, this is exactly what I wanted to avoid and have avoided in the CPU implementation (unnecessarily crunching numbers to only throw them away). The "head" dimensions with MLA are 576 (K) and 512 (V). What the PR does is to use 576 for K and V, and then cuts away the last 64 elements in each row of the FA result. As the multiplication with V with `softmax(K*Q)` is about 2/3 of the total FA computing time (at least on the CPU), this adds a performance penalty of about `2/3*64/512 = 8%`. I'll try a bit more and if I fail, I'll do this for CUDA. There aren't any performance numbers in the PR description. I wouldn't be surprised that this is because performance is lower than just MLA.

---

👤 **davidsyoung** commented the **2025-03-06** at **15:31:11**:<br>

> > This PR from mainline llama.cpp may help with implementing MLA FA [ggml-org/llama.cpp#12227](https://github.com/ggml-org/llama.cpp/pull/12227)
> 
> Ha, this is exactly what I wanted to avoid and have avoided in the CPU implementation (unnecessarily crunching numbers to only throw them away). The "head" dimensions with MLA are 576 (K) and 512 (V). What the PR does is to use 576 for K and V, and then cuts away the last 64 elements in each row of the FA result. As the multiplication with V with `softmax(K*Q)` is about 2/3 of the total FA computing time (at least on the CPU), this adds a performance penalty of about `2/3*64/512 = 8%`. I'll try a bit more and if I fail, I'll do this for CUDA. There aren't any performance numbers in the PR description. I wouldn't be surprised that this is because performance is lower than just MLA.

That makes sense. I did see your current implementation is different than the approach this PR takes. Just said I’d reference it in case it would be useful!

---

👤 **jukofyork** commented the **2025-03-06** at **15:59:38**:<br>

I'd hold off and see what @JohannesGaessler says, as the CUDA version either don't like the "Multi-Query Attention" (MQA) (ie: 1 K/V for 128 Q) and/or the 576 head dimension, as FA is using huge amounts of compute compared to non-FA at the same context...

The non-FA half of the PR might be useful for `ik_llama.cpp`'s `-mla` option though, as I've got rid of all the batched-matrix-multiplies and turned it into just a huge 2D x 2D matrix multiply instead.

---

👤 **jukofyork** commented the **2025-03-06** at **16:01:34**:<br>

> There aren't any performance numbers in the PR description. I wouldn't be surprised that this is because performance is lower than just MLA.

It's running absolutely horrible at long contexts for CUDA - way way worse than these extra 64 values!

---

👤 **ikawrakow** commented the **2025-03-06** at **16:13:32**:<br>

> The non-FA half of the PR might be useful for ik_llama.cpp's -mla option though, as I've got rid of all the batched-matrix-multiplies and turned it into just a huge 2D x 2D matrix multiply instead.

I kept those on purpose. This allows to batch-process `V*softmax(K*Q)` when the context is very large (and no FA is used). Without this ability compute buffers, not KV cache, become the RAM/VRAM limiting factor for very long contexts (and apparently there are many people who would like to use the full 163k context of DeepSeekR1). This is enabled via `-amb value`, whene the value is the maximum size for `K*Q` we want to tolerate in MiB. When this batch processing is not required, my CPU implementation will collapse tensors to lower dimensions if that's advantageous (given the number of heads, tokens, threads). On CUDA things are way more difficult with all the splitting/offloading logic that is mixed up with the compute logic. Hopefully one day @JohannesGaessler will rewrite this stuff so we mere mortals can make changes to the code.

---

👤 **JohannesGaessler** commented the **2025-03-06** at **16:19:24**:<br>

For the split buffers specifically my long-term goal is to move the parallelization logic to the ggml graph level. I intend to do this when optimizing training performance (so probably at some point in the next 12 months). After that the code should become more simpler and easier to work with.

---

👤 **ikawrakow** commented the **2025-03-06** at **16:33:48**:<br>

> so probably at some point in the next 12 months

But people want to run DeepSeek now and not in 12 months :smile:

---

👤 **jukofyork** commented the **2025-03-06** at **17:09:53**:<br>

> This is enabled via `-amb` value, whene the value is the maximum size for K*Q we want to tolerate in MiB.

This looks like a good alternative to reducing memory use if ultimately a head size of 576 isn't feasible. I've currently just been dropping `ubtach-size` as I increase the context, but your `-amb` option would let me keep the larger batch size for everything else.

---

👤 **ikawrakow** commented the **2025-03-06** at **17:48:30**:<br>

> I've currently just been dropping ubatch-size as I increase the context...

This leads to horrible performance for MoE models, especially MoE models such as DoeepSeekV3/R1. Just think about it: the default `u_batch` size is 512, so if you are dropping it, you are using less than that. Say you are using 256. This activates 2048 experts, so each expert has to work on 8 activation rows on average. The performance of such matrix multiplications on CUDA are several times lower (per row) than matrices with 512 or more rows (for the typical LLM model tensor dimensions). If you keep dropping it even further, eventually you are doing GEMVs, so your prompt processing speed starts approaching your TG speed.

---

👤 **davidsyoung** commented the **2025-03-06** at **18:04:26**:<br>

> > This is enabled via `-amb` value, whene the value is the maximum size for K*Q we want to tolerate in MiB.
> 
> This looks like a good alternative to reducing memory use if ultimately a head size of 576 isn't feasible. I've currently just been dropping `ubatch-size` as I increase the context, but your `-amb` option would let me keep the larger batch size for everything else.

For what it’s worth, works incredibly well 

> > This is enabled via `-amb` value, whene the value is the maximum size for K*Q we want to tolerate in MiB.
> 
> This looks like a good alternative to reducing memory use if ultimately a head size of 576 isn't feasible. I've currently just been dropping `ubatch-size` as I increase the context, but your `-amb` option would let me keep the larger batch size for everything else.

For what it’s worth, this works *incredibly well*! 

Can see some generation stats here https://github.com/ikawrakow/ik_llama.cpp/pull/237

---

👤 **jukofyork** commented the **2025-03-06** at **18:12:54**:<br>

> > I've currently just been dropping ubatch-size as I increase the context...
> 
> This leads to horrible performance for MoE models, especially MoE models such as DoeepSeekV3/R1. Just think about it: the default `u_batch` size is 512, so if you are dropping it, you are using less than that. Say you are using 256. This activates 2048 experts, so each expert has to work on 8 activation rows on average. The performance of such matrix multiplications on CUDA are several times lower (per row) than matrices with 512 or more rows (for the typical LLM model tensor dimensions). If you keep dropping it even further, eventually you are doing GEMVs, so your prompt processing speed starts approaching your TG speed.

Yeah, it's not quite as bad for me though as I found that even with `ubatch = 512` the cost of pulling the experts into VRAM over my PCI-E 3x16 bus was slower than just leaving in RAM so I hacked the 32 batch limit for offloading up to something like 9999999 to make it always run on CPU for the non-shared experts (which are the only part not running on GPU due to using the `offload-tensor` option in mainline `llama.cpp`).

This means I only start to see horrible performance drops when I have to drop to a double-digit `ubatch` size (which luckily I don't as I have 96GB VRAM for the oversized compute buffers). I'm still losing some performance compared to what your `-amb` option would give but it's only 10-20% tops due to the lack of CPU compute available with Xeon E5s.

I still like your method better though and agree it is vastly preferable to dropping `ubatch` in the general case!

---

One other thing I've noticed with large contexts and `deepseek-r1` is the use of YaRN and the need for the K-cache to stores pre-RoPEed values, means that as you raise the context length too much; the model starts to get dumber and dumber. For story writing the optimal context length I've found is somewhere between 16k and 32k (4k is pretty bad too, even though that is the pre-YaRN training context).