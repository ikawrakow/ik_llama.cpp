### 🔀 [#94](https://github.com/ikawrakow/ik_llama.cpp/pull/94) - Adding @agray3's graph caching approach

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-18 |
| **Updated** | 2024-10-20 |

---

#### Description

@agray3 has [PR-8366](https://github.com/ggerganov/llama.cpp/pull/8366) open in mainline `llama.cpp` that appears to not meet the high standards of the `llama.cpp` maintainers. Me, being more pragmatic and less of a purist, would like to have these changes here as that way one avoids rebuilding the computation graph for every new token, a "feature" inherited from `llama.cpp` that I don't really like.

Here is what we get in performance improvement on CUDA (RTX-4080 with a Ryzen-7950X CPU)
| model             |       size |     params |          test |    t/s (main)    |       t/s (PR)   |  Speedup |
| ----------------- | ---------: | ---------: | ------------: | ---------------: | ---------------: | -------: |
| llama 8B Q4_0     |   4.33 GiB |     8.03 B |         tg128 |    123.55 ± 0.09 |    125.60 ± 0.11 |  1.017   |   
| llama 3B Q4_0     |   2.08 GiB |     3.61 B |         tg128 |    237.40 ± 1.03 |    244.19 ± 0.71 |  1.029   |   
| llama 1B Q4_0     | 933.24 MiB |     1.50 B |         tg128 |    519.27 ± 2.55 |    538.75 ± 2.32 |  1.038   |   
| llama 2-bpw TriLM |  45.84 MiB |    99.76 M |         tg128 |  1570.51 ± 49.67 |  1754.54 ± 64.75 |  1.117   | 

And here the performance improvement on Metal (M2-Max 30-core GPU, M2-Max CPU):
| model             |       size |          test |      t/s (main)  |    t/s (PR)      |  Speedup |
| ----------------- | ---------: | ------------: | ---------------: | ---------------: | -------: |
| llama 8B Q4_0     |   4.33 GiB |         tg128 |     59.38 ± 0.03 |     60.03 ± 0.03 |  1.011   |
| llama 3B Q4_0     |   2.08 GiB |         tg128 |    107.61 ± 0.55 |    108.74 ± 0.14 |  1.011   |
| llama 1B Q4_0     | 933.24 MiB |         tg128 |    225.92 ± 0.91 |    230.26 ± 0.76 |  1.019   |
| llama 2-bpw TriLM |  45.84 MiB |         tg128 |   520.46 ± 10.70 |    545.46 ± 7.33 |  1.048   |

The speedup obviously increases with decreasing model size as the time computing the graph becomes relatively shorter compared to the time taken building the graph. The speedup I observe is smaller compared to what @agray3 reports in  PR-8366. I guess, it is a matter of how fast the GPU is (where the graph is computed) relative to the CPU (where the graph is built).

GPU performance has not been a focus of this project. Still, how do we do relative to mainline llama.cpp after this PR? Using afd9909a (3942) from today, I get this for the RTX-4080

| model          |       size |          test |        t/s (mainline)|      t/s (PR)    |  Speedup  |
| ---------------| ---------: | ------------: | -------------------: | ---------------: | --------: |
| llama 8B Q4_0  |   4.33 GiB |         tg128 |        122.48 ± 0.10 |    125.60 ± 0.11 |  1.025    |   
| llama 3B Q4_0  |   2.08 GiB |         tg128 |        233.04 ± 0.66 |    244.19 ± 0.71 |  1.048    |   
| llama 1B Q4_0  | 933.24 MiB |         tg128 |        505.63 ± 1.23 |    538.75 ± 2.32 |  1.065    |   
  
and this for the M2-Max


 | model          |       size |          test |      t/s (mainline)  |     t/s (PR)     |  Speedup |
| ---------------| ---------: | ------------: | -------------------: | ---------------: | -------: |
| llama 8B Q4_0  |   4.33 GiB |         tg128 |         57.94 ± 0.32 |     60.03 ± 0.03 |  1.036   |
| llama 3B Q4_0  |   2.08 GiB |         tg128 |        103.67 ± 0.21 |    108.74 ± 0.14 |  1.049   |
| llama 1B Q4_0  | 933.24 MiB |         tg128 |        221.45 ± 1.31 |    230.26 ± 0.76 |  1.039   |


@agray3 Would you review the changes? Alternatively, if you prefer, we can close this PR and you can submit a PR yourself so this contribution is correctly associated with your name.

---

#### 💬 Conversation

👤 **Nexesenex** commented the **2024-10-18** at **17:58:54**:<br>

@ikawrakow : check the "continuation" of this PR also :
https://github.com/ggerganov/llama.cpp/pull/9017

---

👤 **ikawrakow** commented the **2024-10-19** at **09:44:44**:<br>

Oh, btw, 

> @ikawrakow : check the "continuation" of this PR also :
> [ggerganov/llama.cpp#9017](https://github.com/ggerganov/llama.cpp/pull/9017)

Yes, I saw that. But the performance gain there is even less, so not sure if I want to add it.

---

👤 **Nexesenex** commented the **2024-10-19** at **14:07:33**:<br>

Well, IK, little streams make big rivers at some point.
I know you're CPU focused, but as far as I know, only lacks Agray3's missing PR and the MMQ kernels (the "normal" cuda implementation is quite slow and a massive memory hog, and can reach several percents more size occupation of the VRAM for the same model/bbs/ctx) for your new SOTA ggml_types to have the best CUDA inference speed and quality/size reachable in the GGUF ecosystem.

---

👤 **ikawrakow** commented the **2024-10-19** at **14:37:26**:<br>

> only lacks Agray3's missing PR and the MMQ kernels

I know I need to do something about quantized matrix multiplications on CUDA for the new quants. It is not hard to take Johannes' MMQ kernels and adapt. But I have an extremely strong resistance against doing that. I find the MMQ kernels unacceptable, and even less so the several minutes build time associated with them. Adding even more quants will explode build time even further. Each time I want to make a change to one of the headers that I know will trigger full CUDA rebuild, I think 5 times before doing it. I think, a much better approach to pursue there is to find a way to interleave dequantization and matrix multiplications. This is done in the Metal implementation. A simple napkin math shows that the difference in performance between dequantize + cuBLAS matrix multiplication and the MMQ kernels is simply due to the time it takes to store the dequantized tensors in memory. If one would interleave dequantize and matrix multiplications, one would A) (nearly) remove the performance gap B) reduce the extra VRAM required to store the dequantized tensors by a large amount, and C) Get back to normal build times after throwing out the MMQ kernels. I'm just not enough of a CUDA expert to (easily) implements, so keep pushing it out.

---

👤 **agray3** commented the **2024-10-19** at **19:22:56**:<br>

Thanks @ikawrakow. I have now created this PR at https://github.com/ikawrakow/ik_llama.cpp/pull/98 (it is exactly the same as this one). FWIW, to be fair to the llama.cpp maintainers, they are also maintaining the GGML library which can be used separately from llama.cpp and there may be unintended consequences related to that. It should be fine when GGML is always used with llama.cpp.

---

👤 **ikawrakow** commented the **2024-10-20** at **06:36:49**:<br>

Closing in favor of #98