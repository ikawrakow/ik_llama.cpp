### 🔀 [#292](https://github.com/ikawrakow/ik_llama.cpp/pull/292) - Use bf16 instead of fp16 block scales for q8_1

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-26 |
| **Updated** | 2025-03-27 |

---

#### Description

DeepSeek-V3/R1 gives NaNs when inference is run on a computer with `AVX512_VNNI` and the model is quantized with `Q8_0/Q8_0_R8` (issue #285). The difference to vanilla `AVX2` is that in that case activations are quantized with `Q8_1/Q8_1_X4`. The block scale and sum in `Q8_1/Q8_1_X4` are `fp16`.

We did have similar issues with `IQ1_S`, which was solved in #194 by going to a different quantization type for the activations. I did create issue #196 because of that.

We also observed NaNs on CUDA for `IQ4_K` and `IQ4_KS`. These quantization types do not have MMQ kernels, so matrix multiplications were done via dequantization to `fp16` and cuBLAS GEMM. The NaNs were resolved via dequantizing to `bf16` instead (PR #261)

So, it seems one can not use `fp16` arithmetic in DeepSeek-V3/R1.

This is further confirmed by #291, where we observe no NaNs when switching `Q8_0/Q8_0_R8` to vanilla `AVX2` implementation.  

This PR introduces `Q8_2/Q8_2_X4` quantization types that use `bf16` block scale and sum. All quantization types that previously used `Q8_1/Q8_1_X4` to quantize activations for CPU GEMM/GEMV are switched to `Q8_2/Q8_2_X4`.

This should resolve all NaNs on the CPU. 

I wonder why we are not getting NaNs on CUDA for the quantization types that do use `Q8_1`. Or maybe we do, and it is just that nobody has reported.

Closes #285 and #196

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-03-26** at **19:37:47**:<br>

I'm mostly afk until Friday, but will try to rebuild with this PR and test perplexity and imatrix again on a `q8_0` on the CPU only xeon 6980P rig if I get a moment before then. Thanks!

---

👤 **ikawrakow** commented the **2025-03-27** at **04:49:07**:<br>

Thank you for verifying that it works!

---

👤 **saood06** commented the **2025-03-27** at **08:14:07**:<br>

> Closes #285 and #196

This only closed #285, for multiple commands need to use a comma and repeat each command ([source](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue)).

Closes #196

---

👤 **saood06** commented the **2025-03-27** at **08:23:08**:<br>

>So, it seems one can not use fp16 arithmetic in DeepSeek-V3/R1.

Is this why https://github.com/ikawrakow/ik_llama.cpp/discussions/242#discussioncomment-12429240 the imatrix in that comment was failing?

---

👤 **ikawrakow** commented the **2025-03-27** at **08:27:17**:<br>

> Is this why https://github.com/ikawrakow/ik_llama.cpp/discussions/242#discussioncomment-12429240 the imatrix in that comment was failing?

With a very high degree of probability, yes. I get NaNs even for DeepSeek-Lite when I use the `fp16` model on the GPU.