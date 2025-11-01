## 🔀 [Pull Request #608](https://github.com/ikawrakow/ik_llama.cpp/pull/608) - Vulkan: a fresh start

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/vulkan_again` |
| **Target Branch** | `main` |
| **Created** | 2025-07-14 |
| **Updated** | 2025-07-15 |
| **Merged** | 2025-07-15 |

---

## 📄 Description

It looks like something in the Vulkan back-end got broken while porting from mainline and/or adding changes. As I wasn't able to see what could be wrong, I decided to start from scratch from mainline tag `b5891`, and then add the 3 `ik_llama.cpp` fused ops not present in mainline. This PR is the result.

To minimize differences for easier comparisons in the future
* I commented out ops not available in `ik_llama.cpp` instead of deleting them.
* Left in the source tree all the shaders that belong to currently unused ops.
* Tried to minimize the diffs due to back-end interface changes in mainline.

It does seem to work for me, but I would appreciate more comprehensive testing fro @ubergarm, @firecoperana, and others.

Two, I think, interesting observations:
* The Vulkan flash attention implementation absolutely does not work without setting the precision of the op to `fp32`. There is a difference between mainline and `ik_llama.cpp` in that regard. Mainline now just sets the precision to `fp32`, while in `ik_llama.cpp` this is only done for a select set of models. This may have been the actual reason for observing NaNs and gibberish. As I'm not ready to throw in the towel as mainline did at some point, I have changed the attention implementation to set the precision to `fp32` if it is one of the models known to require it, or if the Vulkan backend is enabled. This will have the negative effect of also affecting CUDA, if someone decided to build with CUDA and Vulkan enabled, so probably it would be better  to move this into the Vulkan backend itself (but this is left for a future PR as needed).
* In the previous Vulkan port, I had observed very little difference between `mla = 1` and `mla = 3` (see [#584](https://github.com/ikawrakow/ik_llama.cpp/issues/584)). With this PR I do see, as expected, a significantly higher PP performance with `mla = 3` (e.g., for a context of 16k tokens on an RTX-4080 with coopmat2 enabled, 1470 t/s with `mla = 3` vs 1086 t/s with `mla = 1`.

---

## 💬 Conversation

👤 **ubergarm** commented on **2025-07-14** at **13:34:55**

Thanks again for fussing with the vulkan stuff, so some good news:

## NVIDIA 3090TI
* `KHR_coopmat` - still working perplexity on `Qwen3-14B-Q4_0.gguf` is `Final estimate: PPL = 9.1529 +/- 0.07222`
* `NV_coopmat2` - Its working now with flash attention! Tested multi-turn chat and perplexity of `Qwen3-14B-Q4_0.gguf` comes out clean with `Final estimate: PPL = 9.1502 +/- 0.07219`
```
ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 Ti (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: KHR_coopmat

ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 Ti (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2
```

## AMD 7900 XTX
* `KHR_coopmat` - Working now with flash attention! Tested a multi-turn chat and perplexity of `Qwen3-14B-Q4_0.gguf` comes out clean with `Final estimate: PPL = 9.2161 +/- 0.07294`
```
ggml_vulkan: 0 = Radeon RX 7900 XTX (AMD open-source driver) | uma: 0 | fp16: 1 | warp size: 64 | shared memory: 32768 | int dot: 1 | matrix cores: KHR_coopmat
```

---

👤 **ikawrakow** commented on **2025-07-14** at **13:56:54**

Thanks for testing.

With `KHR_coopmat` and a MoE model (DeepSeek-V2-Lite), I'm hitting an assert when using `u-batch > 512`. It happens with this PR and also in mainline.

---

👤 **ikawrakow** commented on **2025-07-14** at **14:53:24**

Last commit fixes the assert.

---

👤 **firecoperana** commented on **2025-07-14** at **15:02:46**

This PR fixed the issues I have with Vulkan flash attention. Thanks!

---

👤 **ikawrakow** commented on **2025-07-14** at **15:06:51**

Wow, I think this is interesting.

DeepSeek-V2-Lite quantized with `Q4_0`, Vulkan back-end with KHR-coopmat on RTX-4080 GPU. `ik_llama.cpp` is this PR, `llama.cpp` is `55c509daf51d25bfaee9c8b8ce6abff103d4473b` (pulled this morning).

Prompt processing speed

<img width="792" height="612" alt="u12" src="https://github.com/user-attachments/assets/c7038c36-e940-4a68-9a49-9310755dee98" />

Token generation speed

<img width="792" height="612" alt="u13" src="https://github.com/user-attachments/assets/b25492d6-7337-4454-87a2-19e5ea360387" />

Why on Vulkan `mla = 3` is significantly slower than `mla = 1` is something I don't understand at this point as TG is done in exactly the same way (the difference between `mla = 3` and `mla = 1` is in the way prompt processing is performed). 

This is quite a difference in performance, considering that I did nothing other than adding the 3 fused ops.

---

👤 **ikawrakow** commented on **2025-07-14** at **16:47:30**

@jeffbolznv 

You may want to take a look at [this commit](https://github.com/ikawrakow/ik_llama.cpp/pull/608/commits/14ef9ebe9ae45001b778931fcda003ffc1c724a7). Without this change mainline `llama.cpp` hits the assert when using DeepSeel-Lite with `u_batch > 4096/6` and `KHR_coopmat`.

---

👤 **jeffbolznv** commented on **2025-07-14** at **21:45:06**

Thanks, I made a different fix upstream (see https://github.com/ggml-org/llama.cpp/pull/14683).

I noticed FA is failing for the scalar/coopmat1 paths with this model, but working for coopmat2. Did you happen to have a fix for that?

---

👤 **ikawrakow** commented on **2025-07-15** at **05:05:11**

> I noticed FA is failing for the scalar/coopmat1 paths with this model, but working for coopmat2. Did you happen to have a fix for that?

Failing in what sense? I haven't tested scalar, but coopmat1 and coopmt2 seem to be working here.

---

👤 **jeffbolznv** commented on **2025-07-15** at **05:07:34**

I got nonsense output running llama-cli with deepseek and FA enabled. But the backend tests all pass.

---

👤 **ikawrakow** commented on **2025-07-15** at **05:20:56**

I cannot say that I like the responses with coopmat1, but at least it is not gibberish. The above PPL test shows a 0.06 diff between coopmat1 and coopmat2, which is too large to be just numerical roundoff. So, I guess, something is not quite right. I did notice that the Vulkan FA does not work at all with `fp16` precision (one gets NaNs), while using `fp16` arithmetic for self-attention on CUDA is perfectly fine for this model. 

Oh, to answer the question: no, I don't't have a fix.