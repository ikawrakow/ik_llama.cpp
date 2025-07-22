### 🗣️ [#25](https://github.com/ikawrakow/ik_llama.cpp/discussions/25) - CPU prompt processing speed for large contexts

| **Author** | `ikawrakow` |
| :--- | :--- |
| **Created** | 2024-08-22 |
| **Updated** | 2025-01-15 |

---

#### Description

Back in the day when open source / open weight LLMs had a very limited context window, one of the most desired features among LLM enthusiasts was a larger context window. People came up with all sorts of modifications to the RoPE operation, used (LoRA) fine tuning, etc.,  to increase the context window beyond the maximum context used during model training. Today we have open source / open weight models that can handle much longer contexts. E.g., LLaMA-3.1 goes up to 128k tokens, which is probably more than what one can handle with consumer grade hardware for "Inference at the Edge" (and I find it kind of funny to see the many issues opened in the `llama.cpp` repository because users did not limit the maximum context length when running `llama.cpp`, and correspondingly the model would not load because the KV-cache required for 128k tokens does not fit into their <= 24 GB VRAM).

But how well is the large context length being handled?

On the GPU `llama.cpp` has an implementation of Flash Attention (FA), which improves prompt processing speeds for long contexts quite a bit (see the graph below). But, as mentioned, one cannot take advantage of the full context offered by LLaMA-3.1 - me for instance, with the paltry 16 GB VRAM on the RTX-4080 that I have at my disposal, cannot go beyond 32k tokens even for 8B LLaMA-3.1. `llama.cpp` has a FA implementation for the CPU as well, so let's see how well this works:
```
./bin/llama-bench -p 2048 -n 0 -t 16 -fa [0|1] 
```
which gives these results on my Ryzen-7950X CPU:

| model                          |       size |     params | backend    | threads | fa |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -: | ------------: | ---------------: |
| llama 8B Q4_K - Small          |   4.38 GiB |     8.03 B | CPU        |      16 |  0 |     pp2048 |     93.13 ± 0.34 |
| llama 8B Q4_K - Small          |   4.38 GiB |     8.03 B | CPU        |      16 |  1 |        pp2048 |     87.28 ± 0.30 |

Oops. FA is **slower** than no-FA. This is mainline `llama.cpp`. What about the version in this repository where we have much improved CPU prompt processing speed? We get this:

| model                          |       size |     params | backend    | threads | fa |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -: | ------------: | ---------------: |
| llama 8B Q4_K - Small          |   4.38 GiB |     8.03 B | CPU        |      16 |  0 |     pp2048 |     174.09 ± 1.35 |
| llama 8B Q4_K - Small          |   4.38 GiB |     8.03 B | CPU        |      16 |  1 |        pp2048 |    137.87 ± 1.55  |

Oops. Even worse - FA is 26% slower. Why? Because when FA is turned on the `KQ = K * Q` and `KQV = V * KQ` matrix multiplications are handled internally within the FA kernel, so no longer take advantage of the optimized version provided by `iqk_mul_mat`, so performance suffers more.

So, the short answer is: no luck with the current `llama.cpp` version using long contexts on the CPU (unless of course one is very patient).

Anyhow, how well does the CPU do compared to the GPU? The following graph shows the ratio of tokens/second on the CPU to tokens/second on the GPU as a function of prompt length. The CPU is Ryzen-7950X, the GPU is RTX-4080. The black symbols/line is the ratio without GPU Flash Attention, the red circles/line is with FA turned on on the GPU (but not on the CPU). 

![pp_cpu_vs_gpu](https://github.com/user-attachments/assets/9ffb6471-356a-430a-b625-03f4cd1431f0)

The behavior of the curves is interesting for relatively short prompts (say, up to 32 tokens, which is the range of interest for speculative sampling or batch processing), but here we are interested in the portion beyond 500 tokens. Without FA on the GPU, the CPU does improve relative to the GPU with increasing context length, becoming only 16X slower at 32k tokens ("only" considering that we are comparing a $500 previous generation Ryzen to the second fastest consumer grade GPU currently on the market). But when FA is turned on, the performance gap keeps increasing with increasing context length, reaching about 53X slower than the GPU at 32k tokens (and hence the GPU with FA is 3.1X faster compared to no-FA at 32k tokens). 

Clearly it would be useful if we could make the CPU go faster for large contexts.

Here is a quick summary of how the computation time is spent on the CPU when processing a prompt of 32k tokens (using LLaMA-3.1-8B quantized to `Q4_K_S`). For comparison, I have added in the 4th column the fraction of time spent for the various operations in the more "normal" case of processing 512 tokens.

| operation |  time (us)  |  fraction of total time | fraction for PP-512 |
| ---------: | ---: | ---: | ---: |
|               MUL_MAT  | 3.78863e+08  | 0.8022 | 0.9334 |
|              SOFT_MAX | 8.4128e+07  | 0.1781 |  0.0084 |
|              quantize  | 2.32309e+06  | 0.0049 | 0.0159 |
|                   MUL  | 2.117e+06  | 0.0045 | 0.0133 |
|              RMS_NORM | 1.13661e+06  | 0.0024 | 0.0070 |
|                   ADD  | 968962  | 0.0021 | 0.0058 |
|                 SILU |  914848  | 0.0019 | 0.0060 |
|                  ROPE  | 878818  | 0.0019 | 0.0038 |
|                  CONT  | 632398  | 0.0013 | 0.0040 |
|                   CPY  | 306549  | 0.0006 | 0.0021 |
|              GET_ROWS | 12628  | 0.0000 | 0.0002 |

So, basically the entire time is spent doing matrix multiplications and `SOFT_MAX` on the `K*Q` product in the self-attention part (but according to the measured wall time the operation took 495 seconds, while the total of all operations works out to 472 seconds, so there is possibly a ~5% spent on thread synchronization). `SOFT_MAX`, which takes less than 1% of the processing time for 512 tokens increases to 17.8% for a context of 32k. But why is `SOFT_MAX` taking so long? Didn't Justine Tunney just recently contribute a vectorized `expf` implementation to `llama.cpp`, hich should make `SOFT_MAX` go faster? Well, the vectorized `expf` is being used here, but we also need to load from/store back to RAM 2080 GiB while computing `SOFT_MAX`. Given the 84.1 seconds taken by `SOFT_MAX`, this works out to about 25 GiB/s, which is pretty close to the 30 GiB/s the Ryzen-7950X CPU can do in the best case scenario when copying data from here to there.

What about the matrix multiplications? The next table shows total time in us and the fraction of the total matrix multiplication time time taken by the various matrix multiplications (note: this is the sum over all layers):

| Result tensor | Time (us) | Fraction of total time |
| ---: | ---: | ---: |
| kq                  |  1.29016e+08 | 0.3405 |
| kqv                 |  9.59329e+07 | 0.2532 |
| ffn_out             |  4.31925e+07 | 0.1141 |
| ffn_up              |  4.16408e+07 | 0.1099 |
| ffn_gate            |  3.91751e+07 | 0.1034 |
| Qcur                |  1.1825e+07 | 0.0312 |
| kqv_out             |  1.1343e+07 | 0.0299 |
| Vcur                |  3.32323e+06 | 0.0088 |
| Kcur                |  3.29824e+06 | 0.0087 |
| result_output       |  115747 | 0.0003 |

So, close to 60% of the matrix multiplication time is spent for `kq = K*Q` and `kqv = V * softmax(K*Q)`. Combining 60% of 80% with 17.8% for `SOFT_MAX`, we have close to 2/3 of the total time being spent on `K*Q`, `softmax(K*Q)` and `V*softmax(K*Q)`. Interestingly enough, the `kq` and `kqv` matrix multiplications require the exact same amount of floating point operations - 142.94 TFLOP for the 32k context we are looking at. And yet, `kqv` is computed about 35% faster - why? Again, it is a matter of storing data to RAM: `kq` is 2080 GiB (no, we don't keep it all, processing is done in batches), so this works out to 16.1 GiB/s written to memory while computing `kq`. On the other hand `kqv` is "just" 16 GiB, so the matrix multiplication function is storing results at a rate of 0.17 GiB/s - so it is far from being throttled by memory bandwidth. We also see from the data that we get about 1.5 TFLOP/s when computing `kqv`, and about 1.1 TFLOP/s for `kq`. I happen to know that in a synthetic benchmark with just matrix multiplications and result fitting into L2 cache, we get about 2 TFLOP/s with the `iqk_mul_mat` implementation for `fp32`.

Based on this, here are some angles of attack for improving the CPU performance for large prompts:
1. Investigate if it is possible to get the `kqv` speed closer to the 2 TFLOP/s we know is achievable
2. Investigate if we can improve `kq` performance by better interleaving computation with memory writes. We are at ~16 GiB/s and 30 GiB/s is the limit on this CPU
3. Fuse `kq` and `softmax(kq)` into a single operation. As I don't want to go implement this new operation on all back-ends, the fusing should be done on-the-fly while evaluating the computation graph on the CPU. This will eliminate writing `kq` to RAM, so has the potential of shaving off at least 15% of the time
4. Fuse `K*Q`, `softmax(K*Q)` and `V*softmax(K*Q)` into a single operation. I.e., re-discover Flash Attention :-) As the experience with the `llama.cpp` CPU implementation shows, it is not just a matter of not storing intermediate results into RAM. One still needs to go as fast as possible with the matrix multiplications to actually get performance improvement from this.
5. Look into quantized KV cache. Quantized matrix multiplications are faster than `fp32` - we get in the 2.5 to 3 TFLOP/s with the implementation in `iqk_mul_mat`, but I need to look in more detail into the associated accuracy loss. In addition, if `V` is quantized, `softmax(K*Q)` must be quantized as well, which may be too costly unless fused into the `softmax(K*Q)` operation.

---

#### 🗣️ Discussion

👤 **jart** replied the **2024-08-22** at **15:26:07**:<br>

> ~5% spent on thread synchronization

Have you tried these measurements with the latest llamafile sources? There's a variety of improvements to thread synchronization. For example, here's a better memory barrier that's more on par with what GNU OpenMP does.

```c
void ggml_barrier(const struct ggml_compute_params * params) {
    if (params->shared->n_threads == 1)
        return;
    int n = params->shared->n_threads;
    atomic_int * count = &params->shared->n_barrier;
    atomic_uint * phase = &params->shared->n_barrier_passed[params->ith].i;
    unsigned i = atomic_load_explicit(phase, memory_order_relaxed);
    if (atomic_fetch_add_explicit(count, 1, memory_order_acq_rel) == n - 1) {
        atomic_store_explicit(count, 0, memory_order_relaxed);
        for (int j = 0; j < n; ++j)
            atomic_store_explicit(&params->shared->n_barrier_passed[j].i,
                                  i + 1, memory_order_relaxed);
        atomic_thread_fence(memory_order_release);
    } else {
        while (atomic_load_explicit(phase, memory_order_relaxed) == i)
            pthread_pause_np();
        atomic_thread_fence(memory_order_acquire);
    }
}
```

In `ggml_graph_compute_thread()` it helps a lot to say:

```c
    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        if (ggml_is_noop(node->op)) // [jart]
            continue;

        // ...
```

Assuming you have this defined:

```c
static bool ggml_is_noop(enum ggml_op op) { // [jart]
    switch (op) {
        case GGML_OP_NONE:
        case GGML_OP_PERMUTE:
        case GGML_OP_RESHAPE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_VIEW:
            return true;
        default:
            return false;
    }
}
```

llama.cpp also likes to spawn a thread for every token when predicting. You can make threads spawn/join 10x faster with this:

- https://github.com/Mozilla-Ocho/llamafile/blob/main/llamafile/pool.cpp

Is this all something that'd interest you? I can easily send a PR adding it to your repo if you don't care about things like MSVC.

---

👤 **ikawrakow** replied the **2024-08-22** at **16:16:08**:<br>

Hey @jart, thanks for the comments!

> Have you tried these measurements with the latest llamafile sources? There's a variety of improvements to thread synchronization. For example, here's a better memory barrier that's more on par with what GNU OpenMP does.

No, I'm working with my `llama.cpp` clone and using OpenMP on Linux. On my M2-Max OpenMP is somehow really bad, so I'm using a slightly modified version of `ggml_barrier`, see [here](https://github.com/ikawrakow/ik_llama.cpp/blob/bd99ed7d0afd2b12c0f5ff5c17b58486396dfe7e/ggml/src/ggml.c#L3371). But I'll definitely look into using threads differently. It hasn't been an issue with my setup until I started looking into these long contexts. When you do long contexts the computation takes quite some time, so the OS will definitely preempt one or more threads at some point, and then we end up waiting for them to finish with the `ggml` approach of splitting the work into `n_thread` chunks. I think for the long contexts it will be better to do work stealing from a pool of tasks that is a few times larger than the number of threads. I'm planning to also look into that.

> In ggml_graph_compute_thread() it helps a lot to say:  

Ha, you had already done that! I didn't check `llamafile` and discovered this on my own, see [this PR](https://github.com/ikawrakow/ik_llama.cpp/pull/19)

> Is this all something that'd interest you? I can easily send a PR adding it to your repo if you don't care about things like MSVC.

I don't care about MSVC, so sure. There is the MIT vs Apache-2.0 issue, but we can sort that out.

> 👤 **jart** replied the **2024-08-22** at **18:02:15**:<br>
> Apple doesn't have OpenMP. So that's where my thread synchronization changes have the most impact. Right now in llama.cpp if I build it on my Apple M2 and run with `-ngl 0` for CPU mode it gets 134 tok/sec tops. But llamafile with `-ngl 0` on MacOS M2 generates text at anywhere from 150 tok/sec to 210 tok/sec depending on how much Netflix is interfering and how much I win the XNU scheduler lottery (I imagine things are consistently 200+ if Asahi Linux is used instead of XNU). On the other hand, if I use Metal GPU then it consistently generates text at 200 tok/sec.
> 
> Yes, that's correct. I'm claiming that the changes you and I both made on llamafile have made M2 Ultra CPU go faster than its GPU sometimes when generating text with TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf. However if I use a larger model like Mistral 7b where the matmuls start to dominate a lot more than the sync barriers, then I can only generate 42 tok/sec and GPU does 72 tok/sec. So this is all a bit orthogonal to the goal here of huge context windows. I just wanted you to know that we did something most people would likely assume is not possible. I certainly wouldn't have, because when I started focusing on this in January I set out with the goal of making CPU at at least only 10x slower than GPU.
> 
> 👤 **jart** replied the **2024-08-22** at **18:13:48**:<br>
> As for MIT vs. Apache 2.0 there's a lot of leeway from Mozilla to make my work available to other local AI projects under the MIT license if that's what you're using here. I'll roll up a pull request for you sometime in the next few days, that'll work smoothly on POSIX platforms.
> 
> 👤 **ikawrakow** replied the **2024-08-22** at **19:08:09**:<br>
> > Apple doesn't have OpenMP
> 
> I thought the currently recommended approach in `llama.cpp` is to `brew install libomp`, which then by default enables OpenMP? That's what I tried anyway after observing a horrible performance with the `ggml_barrier` implementation on my M2-Max laptop, but that didn't help much either, so I did end up putting in the inline assembly that fixed performance for me.
> 
> But yes, for small models such as TinyLlama  thread synchronization becomes really important, so I should try your barrier version.
> 
> 👤 **jart** replied the **2024-08-22** at **22:12:59**:<br>
> I don't even know why OpenMP is there. It's a GPL-licensed library. We might as well be using Torch if we're going to link that. Goes against the very spirit of the project which is figuring these things out for ourselves.
> 
> 👤 **jart** replied the **2024-08-22** at **22:16:45**:<br>
> Also if by libomp you mean LLVM libomp, sadly it's kind of an newer alternative and it's got none of the alpha of GNU's OpenMP runtime. Based on my own evaluation, LLVM libomp is about as fast as llama.cpp's old synchronization code, when it's applied for GGML speedups.

---

👤 **ikawrakow** replied the **2024-08-27** at **06:31:49**:<br>

I did try a few things on [this branch](https://github.com/ikawrakow/ik_llama.cpp/tree/ik/kq_fused_softmax), but nothing is really working. The branch is just exploratory, absolutely not production ready, and `AVX512`-only. Given the unsatisfactory outcome, it will not get merged.
* I can get the CPU flash attention to run faster than the original (quite a bit faster for very large prompts), but it is still slower than no flash attention
* I can get a ~3% speedup for large prompts by optimizing for no-alibi and causal attention mask. But given the marginal improvement, increased complexity, and reduced generality, it does not seem worth adding.

On the bright side, PR #27 merges "soft-capping" with soft-max. For large prompts, this leads to a significant performance boost for Gemma-2 models. At 32k tokens and Gemma-2-2b, the performance gap between GPU with flash attention and the Ryzen-7950X CPU is now "only" a factor of 45 (instead of the 53X in the above graph).

---

👤 **ikawrakow** replied the **2024-08-30** at **15:25:30**:<br>

OK, I have progress on [this branch](https://github.com/ikawrakow/ik_llama.cpp/tree/ik/kq_fused_softmax). Extremely hacky and `AVX512`-only (or, more precisely, Zen4-only), totally not production ready. But I'm finally able to outperform no flash attention on my Ryzen-7950X CPU - by about 20% for context of 16k, 23% for 32k, with LLaMA-3.1-8B.

This graph shows the current status. y-axis is tokens per second on my Ryzen-7950X CPU, x-axis is context size (logarithmic scale). Black symbols show the performance in this repository, green is mainline `llama.cpp`, both without FA. The red symbols is what we get if we turn on FA as inherited from `llama.cpp`, so complete disaster. Blue symbols are mainline `llama.cpp` with FA. Yes, it is slower than no-FA (and the fact that it is slower on most platforms except newer GPU's with CUDA appears to be not well known). The magenta symbols show the results for the new FA implementation on the [ik/kq_fused_softmax](https://github.com/ikawrakow/ik_llama.cpp/tree/ik/kq_fused_softmax) branch. There are many attempts there, so this is the result of [this function](https://github.com/ikawrakow/ik_llama.cpp/blob/77b7baaff79cdc94fc13bd67698e85a40a55bb00/ggml/src/iqk/iqk_mul_mat.cpp#L6786)

![fa](https://github.com/user-attachments/assets/4f5b7e7a-0648-4972-ba93-cd14da3ab1e6)

My guess is that there is still a bottleneck at 32k tokens. Based on the FA to n-FA relative performance increase up to 16k tokens I would expect a performance gain above 30% at 32k tokens instead of the 23% we currently get.

---

👤 **ikawrakow** replied the **2024-08-30** at **15:37:24**:<br>

And here is how the raltive CPU vs GPU performance graph changes with the new CPU flash attention implementation. The FA curve is basically flat now beyond 1000 tokens, except at 32k where I suspect a bottleneck that I have not found. 

 
![pp_cpu_vs_gpu](https://github.com/user-attachments/assets/96c27976-f22b-4fa9-a0b5-021f0992a83c)

---

👤 **ikawrakow** replied the **2025-01-15** at **17:50:21**:<br>

There has been progress since I last wrote here, with PR #172 being the latest contribution to improving CPU prompt processing speed. The following graph is for LLaMA-3.1-8B-Instruct quantized to `IQ4_XS` (which seems a fairly popular quantization type). Tested on a Ryzen-7950X CPU. The mandatory current mainline `llama.cpp` results are for `build: 1d850433 (4488)`. The results for `ik_llama.cpp` are obtained using run-time-repacking to the corresponding 4-row interleaved variant.

![pp512_vs_ctx](https://github.com/user-attachments/assets/81a09390-b0da-4d5c-9815-300b4b86705c)

* In mainline `llama.cpp` FA continues to be underwhelming, being handsomely outperformed by not using FA
* `ik_llama.cpp` now finally exceeds 100 t/s for a prompt of 32k tokens. I get 122 t/s (`BF16` KV-cache) and 113 t/s (`Q8_0` KV-cache). The best I could do with mainline is 37 t/s (`Q8_0` K-cache, no FA).
* I'm quite pleased that `Q8_0` KV-cache is now almost on par with `BF16`
* `ik_llama.cpp` is almost 4 times faster than mainline at 256 tokens, and still 3.3 times faster at 32k tokens. For such large contexts the computation time is heavily dominated by the `K*Q` and `V*softmax(K*Q)` matrix multiplications, with these matrices by far exceeding L3 cache size, and hence the operation becoming memory bound. In fact, part of the improvement in PR #172 is due to reducing the number of memory loads from the `V`-cache in the FA computation.
* If processing very long context is a significant use case, utilizing `Q8_K_R8` brings additional gains. We get 373 t/s for 512 tokens, 312 t/s at 4k, 268 t/s at 8k, 203 t/s at 16k, and 136 t/s at 32k tokens.

It is also interesting to look at the performance relative to a GPU. I'm using an RTX-4080 GPU with the same model and FA enabled. Compared to earlier plots in this thread, I have changed the plot to show the ratio of GPU to CPU prompt processing speed and have restricted the prompt length to $\ge 100$ tokens to reduce the range of the y-axis.  The Ryzen-7950X now saturates at about 27.5X lower performance compared to the RTX-4080, which is not bad at all.

![pp_gpu_vs_cpu](https://github.com/user-attachments/assets/ef674c0e-7556-4bbe-96cb-658a530aabc6)