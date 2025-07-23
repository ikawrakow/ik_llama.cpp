### ✨ [#255](https://github.com/ikawrakow/ik_llama.cpp/issues/255) - Feature Request: dynamic layer by layer offloading during prompt processing for VRAM constrained scenarios

| **Author** | `binjiechen` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-13 |
| **Updated** | 2025-03-15 |

---

#### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

During prompt processing (possibly long context), allow dynamically layer by layer offload instead of fixed offload. i.e., offload layer 1 to GPU, process a batch of tokens, then free layer 1 and offload layer 2 to GPU, ...
A large batch can be used and compute buffers can be freed before token generation. Optionally, some layers can be retained if VRAM is large enough. It should only work for parallel = 1 I guess.

### Motivation

From my experience, prompt processing stage is compute bound as usually a large batch size is used.
When VRAM < model size, only a part of the model can be offloaded to GPU and the CPU part could be bottleneck. So, if we offload layer by layer, GPU can be fully utilized and can offer better performance.

I have a 4090 and a 13600k (power limited to 125w) and 192GB memory. I ran some tests on Qwen 2.5 32B which has 64 blocks:
| model                          |       size |     params | backend    | ngl | threads | n_ubatch | fa | mmap | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | ---: | ---: | ------------: | ---------------: |
| qwen2 ?B Q5_K - Medium         |  21.66 GiB |    32.76 B | CUDA       |  64 |      13 |     2048 |  1 |    0 |    1 |        pp2048 |   2627.90 ± 0.00 |
| qwen2 ?B Q5_K - Medium         |  21.66 GiB |    32.76 B | CUDA       |  63 |      13 |     2048 |  1 |    0 |    1 |        pp2048 |    572.61 ± 0.00 |
| qwen2 ?B Q5_K - Medium         |  21.66 GiB |    32.76 B | CUDA       |  60 |      13 |     2048 |  1 |    0 |    1 |        pp2048 |    173.71 ± 0.00 |
| qwen2 ?B Q5_K - Medium         |  21.66 GiB |    32.76 B | CUDA       |  40 |      13 |     2048 |  1 |    0 |    1 |        pp2048 |     30.66 ± 0.00 |
| qwen2 ?B Q5_K - Medium         |  21.66 GiB |    32.76 B | CUDA       |  20 |      13 |     2048 |  1 |    0 |    1 |        pp2048 |     16.93 ± 0.00 |
| qwen2 ?B Q5_K - Medium         |  21.66 GiB |    32.76 B | CUDA       |   0 |      13 |     2048 |  1 |    0 |    1 |        pp2048 |     10.76 ± 0.00 |

Even if only 1 block is left on CPU, t/s is decreased by 78%. I think layer by layer offloading should help a lot in this situation. Assume a 8GB/s RAM to VRAM speed, then the whole offloading would only cost 2.7s in this case and result in a speed of 758 t/s (if compute is hidden by transfer).

### Possible Implementation

_No response_

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-03-14** at **08:13:39**:<br>

I can look into this next week (travelling right now). But I think there maybe something wrong with the offloading to the GPU. I don't think not offloaded layers are run on the CPU. To check, build without CUDA and run the same benchmarks. I expect performance much better than what you observe with zero layers offloaded.

---

👤 **binjiechen** commented the **2025-03-14** at **10:42:34**:<br>

> I can look into this next week (travelling right now). But I think there maybe something wrong with the offloading to the GPU. I don't think not offloaded layers are run on the CPU. To check, build without CUDA and run the same benchmarks. I expect performance much better than what you observe with zero layers offloaded.

The result with a CPU only build is basically the same:
| model                          |       size |     params | backend    | threads | n_ubatch | fa | mmap | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | -: | ---: | ---: | ------------: | ---------------: |
| qwen2 ?B Q5_K - Medium         |  21.66 GiB |    32.76 B | BLAS       |      13 |     2048 |  1 |    0 |    1 |        pp2048 |     11.79 ± 0.00 |

I think not offloaded layers are indeed run on the CPU, as in the previous test with CUDA backend, I observe full CPU utilization from htop.
Anyway, thanks for your great work and enjoy your travelling!

---

👤 **ikawrakow** commented the **2025-03-15** at **08:32:57**:<br>

Very strange. My GPU is RTX-4080, so I can fit a maximum of 45 layers on the GPU for 32B Qwen2.5, and here is what I get with that:

| model                          |       size |     params | backend    | ngl | threads | n_ubatch |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | ------------: | ---------------: |
| qwen2 ?B Q4_K - Medium         |  18.50 GiB |    32.76 B | CUDA       |  45 |      16 |     2048 |        pp2048 |  1030.55 ± 11.82 |
| qwen2 ?B Q4_K - Medium         |  18.50 GiB |    32.76 B | CUDA       |  40 |      16 |     2048 |        pp2048 |    985.30 ± 2.18 |
| qwen2 ?B Q4_K - Medium         |  18.50 GiB |    32.76 B | CUDA       |  20 |      16 |     2048 |        pp2048 |    817.11 ± 1.09 |
| qwen2 ?B Q4_K - Medium         |  18.50 GiB |    32.76 B | CUDA       |  10 |      16 |     2048 |        pp2048 |    750.98 ± 0.70 |
| qwen2 ?B Q4_K - Medium         |  18.50 GiB |    32.76 B | CUDA       |   0 |      16 |     2048 |        pp2048 |   703.04 ± 16.27 |
| qwen2 ?B Q4_K - Medium         |  18.50 GiB |    32.76 B | CPU        |   0 |      32 |     2048 |        pp2048 |     40.63 ± 0.55 |

The last line in the table is with a CPU-only build, the other line with zero layers offloaded is the CUDA build. So, clearly the model gets offloaded to the GPU for the actual computation. Performance with the 0 layers offloaded is ~70% of the performance with 45 layers offloaded. When 45 layers are offloaded, computing a batch of 2048 tokens takes about 2 seconds. With zero layers offloaded it is 2.9 seconds. So, offloading takes 0.9 seconds. 45 layers are `45/64*18.5 =13 GiB`, so we can estimate the throughput of the PCI-E transfer to be `13/0.9=14.4 GiB/s`, pretty much in line with the expectation.

It would seem that in your case the layers do not get offloaded to the GPU for some reason. What is the exact model you are using?

Btw, the current multi-threading here (and also upstream) is not very good for CPUs with performance and efficiency cores. The work get simple split in `n_thread` equal chunks, so the duration of each operation is determined by the performance of the efficiency cores. Have you tried using just the P cores?

---

👤 **ikawrakow** commented the **2025-03-15** at **11:06:05**:<br>

Aha, I know where the problem is. Try disabling BLAS. I never enable it because the `iqk_mul_mat` matrix multiplications are faster than any CPU `BLAS` implementation I have tried.

What happens with BLAS enabled is this: the scheduler goes through all back-ends and checks if they support the operation being scheduled. If more than one back-end is found that supports the operation, then the operation is scheduled on the back-end that already has the model weights participating in the op. Hence, with BLAS enabled (another back-end), matrix multiplications for not offloaded layers get scheduled on the BLAS back-end, and hence they run on the CPU.

---

👤 **binjiechen** commented the **2025-03-15** at **12:58:47**:<br>

> Aha, I know where the problem is. Try disabling BLAS. I never enable it because the `iqk_mul_mat` matrix multiplications are faster than any CPU `BLAS` implementation I have tried.
> 
> What happens with BLAS enabled is this: the scheduler goes through all back-ends and checks if they support the operation being scheduled. If more than one back-end is found that supports the operation, then the operation is scheduled on the back-end that already has the model weights participating in the op. Hence, with BLAS enabled (another back-end), matrix multiplications for not offloaded layers get scheduled on the BLAS back-end, and hence they run on the CPU.

Ah, yes, I resolved the problem. I now have a better understanding of how llama.cpp works. Thank you very much!
| model                          |       size |     params | backend    | ngl | threads | n_ubatch | fa | mmap | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | ---: | ---: | ------------: | ---------------: |
| qwen2 ?B Q5_K - Medium         |  21.66 GiB |    32.76 B | CUDA       |  64 |       6 |     2048 |  1 |    0 |    1 |        pp2048 |   2657.76 ± 0.00 |
| qwen2 ?B Q5_K - Medium         |  21.66 GiB |    32.76 B | CUDA       |  32 |       6 |     2048 |  1 |    0 |    1 |        pp2048 |   1622.08 ± 0.00 |
| qwen2 ?B Q5_K - Medium         |  21.66 GiB |    32.76 B | CUDA       |   0 |       6 |     2048 |  1 |    0 |    1 |        pp2048 |   1161.20 ± 0.00 |

Results looks really great this time.

| model                          |       size |     params | backend    | threads | n_ubatch | fa | mmap | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | -: | ---: | ---: | ------------: | ---------------: |
| qwen2 ?B Q5_K - Medium         |  21.66 GiB |    32.76 B | CPU        |      13 |     2048 |  1 |    0 |    1 |        pp2048 |     10.04 ± 0.00 |
| qwen2 ?B Q5_K - Medium         |  21.66 GiB |    32.76 B | CPU        |       8 |     2048 |  1 |    0 |    1 |        pp2048 |      8.32 ± 0.00 |
| qwen2 ?B Q5_K - Medium         |  21.66 GiB |    32.76 B | CPU        |       6 |     2048 |  1 |    0 |    1 |        pp2048 |     11.46 ± 0.00 |
| qwen2 ?B Q5_K - Medium         |  21.66 GiB |    32.76 B | BLAS       |       6 |     2048 |  1 |    0 |    1 |        pp2048 |     11.84 ± 0.00 |

For CPU backend, it's true that using only P cores gives better performance. Intel oneMKL BLAS is slightly faster under this setting

| model                          |       size |     params | backend    | ngl | threads | n_ubatch | fa | mmap | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -: | ---: | ---: | ------------: | ---------------: |
| qwen2 ?B Q5_K - Medium         |  21.66 GiB |    32.76 B | CUDA       |  64 |       6 |     2048 |  1 |    0 |    1 |         tg128 |     25.23 ± 0.00 |
| qwen2 ?B Q5_K - Medium         |  21.66 GiB |    32.76 B | CUDA+BLAS       |  64 |       6 |     2048 |  1 |    0 |    1 |         tg128 |     25.60 ± 0.00 |
| qwen2 ?B Q5_K - Medium         |  21.66 GiB |    32.76 B | CUDA       |  32 |       6 |     2048 |  1 |    0 |    1 |         tg128 |      4.00 ± 0.00 |
| qwen2 ?B Q5_K - Medium         |  21.66 GiB |    32.76 B | CUDA+BLAS       |  32 |       6 |     2048 |  1 |    0 |    1 |         tg128 |      4.45 ± 0.00 |
| qwen2 ?B Q5_K - Medium         |  21.66 GiB |    32.76 B | CUDA       |   0 |       6 |     2048 |  1 |    0 |    1 |         tg128 |      2.18 ± 0.00 |
| qwen2 ?B Q5_K - Medium         |  21.66 GiB |    32.76 B | CUDA+BLAS       |   0 |       6 |     2048 |  1 |    0 |    1 |         tg128 |      2.43 ± 0.00 |

Also, I find that for token generation which seems memory bound, CUDA+BLAS gives better performance (for ngl > 64 they're the same). So is it possible to add an option that makes CPU a valid backend and do the computation during token generation?

---

👤 **ikawrakow** commented the **2025-03-15** at **13:19:12**:<br>

What happens if you add `-rtr 1`? Is one MKL still faster for CPU-only PP?

---

👤 **binjiechen** commented the **2025-03-15** at **13:48:19**:<br>

| model                          |       size |     params | backend    | threads | n_ubatch | fa | mmap | rtr | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | -: | ---: | --: | ---: | ------------: | ---------------: |
| qwen2 ?B Q5_K - Medium         |  21.66 GiB |    32.76 B | CPU        |       6 |     2048 |  1 |    0 |   1 |    1 |        pp2048 |     16.96 ± 0.00 |
| qwen2 ?B Q5_K - Medium         |  21.66 GiB |    32.76 B | BLAS       |       6 |     2048 |  1 |    0 |   1 |    1 |        pp2048 |     11.78 ± 0.00 |

With `-rtr 1`, BLAS version is not affected and non-BLAS is significantly faster.

---

👤 **ikawrakow** commented the **2025-03-15** at **16:07:08**:<br>

In that case, is there a reason to use BLAS? Your TG benchmark shows a slightly better TG performance with BLAS, but I don't really understand why that would be the case. For TG matrix multiplications are not done by BLAS even if it is enabled.

---

👤 **binjiechen** commented the **2025-03-15** at **16:43:12**:<br>

> In that case, is there a reason to use BLAS? Your TG benchmark shows a slightly better TG performance with BLAS, but I don't really understand why that would be the case. For TG matrix multiplications are not done by BLAS even if it is enabled.

No, BLAS is not needed. I thought during TG not offloaded layers are also computed on GPU, and what I meant previously is to keep computation on CPU for not offloaded layers. So weight transfer does not happen, which might increase performance.

But I'm confused now, when ngl is 0, if all computation is on GPU, then TG speed shouldn't be as high as 2 t/s. So during TG, not offloaded layers' computation is actually done on CPU?

---

👤 **ikawrakow** commented the **2025-03-15** at **16:48:25**:<br>

> So during TG, not offloaded layers' computation is actually done on CPU?

Yes. There is a magic threshold set in the CUDA back-end (currently 32). If the batch size is less than that, tensors are not offloaded to the GPU, and the calculation is done on the CPU. One can try to be more intelligent and make it dependent on amount of data that needs to be uploaded, PCI-E speed, relative CPU vs GPU matrix multiplication performance, etc. But for now that's what it is.

---

👤 **binjiechen** commented the **2025-03-15** at **17:00:25**:<br>

Ok, I got it now. Thanks for your patience!