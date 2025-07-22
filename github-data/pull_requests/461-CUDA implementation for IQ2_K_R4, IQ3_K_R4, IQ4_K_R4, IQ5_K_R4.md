### 🔀 [#461](https://github.com/ikawrakow/ik_llama.cpp/pull/461) - CUDA implementation for IQ2_K_R4, IQ3_K_R4, IQ4_K_R4, IQ5_K_R4

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-26 |
| **Updated** | 2025-06-04 |

---

#### Description

The `IQX_K` quants and their row-interleaved siblings `IQX_K_R4` offer better quantization quality than corresponding i-, k-, or legacy quants at the same bpw. `IQX_K_R4` quants have better CPU performance but cannot be used on CUDA as there is no GEMM/GEMV implementation. Hence, "quant cookers" need to release `IQX_K` quantized model, so users can use them on their GPUs, but that requires users doing CPU-ony inference to repack the model to take advantage of the better CPU performance. In addition, @ubergarm has released various `IQK_X_R4` quantized models (see [here](https://huggingface.co/ubergarm), and those cannot be used for GPU inference. 

To remove such inconvenience, this PR adds CUDA implementation for the row-interleaved quants `IQ2_K_R4, IQ3_K_R4, IQ4_K_R4, IQ5_K_R4`. I'll follow up with a separate PR for `IQ2_KS_R4, IQ4_KS_R4` and `IQ5_KS_R4`.

For now GEMM is implemented via dequantize + cuBLAS. I may add quantized GEMM (a.k.a. MMQ) later.

**Note**: because of the above, if you want to use a `IQX_K_R4` DeepSeek-V3/R1 model on the GPU, you may need to build with `-DGGML_CUDA_IQK_FORCE_BF16=1` to force `bf16` arithmetic with cuBLAS as `fp16` has been noted to lead to numerical instabilities and garbled output. I did not enable `GGML_CUDA_IQK_FORCE_BF16` by default as it reduces prompt processing performance while, as far as I can tell, `bf16` is only required for DeepSeek.

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-05-30** at **15:22:27**:<br>

>  I'll follow up with a separate PR for IQ2_KS_R4, IQ4_KS_R4 and IQ5_KS_R4.

I was looking to use the `IQ2_KS_R4` type for a smaller `R1-0528` quant, but noticed it isn't implemented afaict:

```bash
$ grep repacked examples/quantize/quantize.cpp | grep KS
    { "IQ4_KS_R4",LLAMA_FTYPE_MOSTLY_IQ4_KS_R4,"IQ4_KS repacked", },
    { "IQ5_KS_R4",LLAMA_FTYPE_MOSTLY_IQ5_KS_R4,"IQ5_KS repacked", },

$ grep KS_R4 include/llama.h
        LLAMA_FTYPE_MOSTLY_IQ4_KS_R4     = 345, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ5_KS_R4     = 350, // except 1d tensors

$ grep KS_R4 ggml/src/ggml-cuda/convert.cu
        case GGML_TYPE_IQ4_KS_R4:
        case GGML_TYPE_IQ5_KS_R4:
        case GGML_TYPE_IQ4_KS_R4:
        case GGML_TYPE_IQ5_KS_R4:
        case GGML_TYPE_IQ4_KS_R4:
        case GGML_TYPE_IQ5_KS_R4:
```

For now I'll go with `IQ3_K_R4` and `IQ2_K_R4`. I might loop back in the future if you decide to implement `IQ3_KS_R4` and `IQ2_KS_R4` which presumably could be a little faster and useful for these big DeepSeek models. No pressure and thanks again for your patience as I try to keep up with everything! Cheers!

---

👤 **ikawrakow** commented the **2025-05-30** at **15:35:19**:<br>

No I haven't done `IQ2_KS_R4` yet. I keep trying to improve it, so I got distracted with that. And, because there isn't much usage of it yet, I was considering making a breaking change to the packing. That was the actual reason for postponing the CUDA implementation. 

Perhaps just use `iq2_k_r4` for now?

Or, if you have the patience to wait for `iq2_kt`, you can try quantizing the `ffn_up` and `ffn_gate` tensors with that. It is slightly less bpw than `iq2_ks` (2.125 vs 2.1875), but you get lower PPL. CUDA and Metal performance are quite decent. The downside is that CPU performance is pretty bad.

---

👤 **ubergarm** commented the **2025-06-01** at **15:28:53**:<br>

> I did not enable GGML_CUDA_IQK_FORCE_BF16 by default as it reduces prompt processing performance while, as far as I can tell, bf16 is only required for DeepSeek.

I got a report from the wild that FORCE_BF16=1 gave a speed boost and confirmed that it does seem to do so at least in this specific hardware configuration and this specific quant. I added a graph and data to the R1-0528 discussion: https://github.com/ikawrakow/ik_llama.cpp/discussions/477#discussioncomment-13335019

> Or, if you have the patience to wait for iq2_kt, you can try quantizing the ffn_up and ffn_gate tensors with that. It is slightly less bpw than iq2_ks (2.125 vs 2.1875), but you get lower PPL. 

OOOH! I just realized you've been doing the `iqN_kt` "trellis quants" which are the QTIP/exl3 quants for a while. I can be quite myopic. Reading through some old PRs I see you've done quite a bit already. I've been impressed by the low perplexity (especially with such low 2~3 bpw) using exllamav3 to make exl3 quants following @louiehelm 's quest for the best magic number e.g. `3INST mcg=0xB83EA16`

![plot-kld-Qwen3-30B-A3B-exl3](https://github.com/user-attachments/assets/f9127d6f-56a7-4c9f-9d07-97bfa335a0bb)

I wish I had a way to compare apples-apples between exl3 and ik_llama.cpp but as there is no llama-cpp-python bindings for ik_llama.cpp. (i tried for half an hour to get it to work with older versions but things had diverged too much already a year ago so gave up).

Regardless, I'll read up more on your implementation of iq2_kt and check the code for the mcg value etc. Thanks!

---

👤 **ikawrakow** commented the **2025-06-01** at **15:57:38**:<br>

> OOOH! I just realized you've been doing the iqN_kt "trellis quants" which are the QTIP/exl3 quants for a while. I can be quite myopic. Reading through some old PRs I see you've done quite a bit already. I've been impressed by the low perplexity (especially with such low 2~3 bpw) using exllamav3 to make exl3 quants following @louiehelm 's https://github.com/turboderp-org/exllamav3/pull/26#issuecomment-2916801280 e.g. 3INST mcg=0xB83EA16

The `IQX_KT` quants can be used right now with very decent performance on CUDA. The patience is not required to wait for me to finish working on them, but to have the patience to wait for the quantization to finish. Quantization of those is ~5X slower than `IQK` quants. 

On the CPU performance is not quite as good. PP performance is getting there, but TG is slooow on the CPU.

I did look a bit into the plots in the ExLlamaV3 repository. I absolutely cannot confirm the PPL plots for LLaMA-3-70B. I used the 70B model because in my experience when overfitting is going on, the overfitting is typically based on the small models (nobody has the patience to fool around with meta parameters with testing done on a large model). Hence, color me skeptical about the ExLlamaV3 results.

The thing about apples-to-apples is that if you use `PPL(Q)/PPL(f16)` (or better, `ln(PPL(Q)/PPL(f16))`, which is directly related to KLD), you will find that it is nearly independent of the way PPL has been calculated (for the same test corpus). That allows you to make apples-to-apples comparisons while having apples and oranges.

---

👤 **louiehelm** commented the **2025-06-02** at **04:14:53**:<br>

I like KT quants too and tried subbing out 3INST parameters with superior ones (since LCG from QTIP paper x = 89226354 * x + 64248484 can't be optimal) but for some reason, all the better parameters with lower MSE both in synthetic trellis codes (without rotations) or in EXL3 (with rotations) don't show improvement when I slot them into ik_llama, recompile, quant, and test models.

Could current KT code paths be implicitly tuned to expect certain behavior the default parameters provide? I haven't gone through the code super carefully but at first glance I can't immediately figure this out.

I've found dozens of better decoder params for 3INST that show ~5% reduction in MSE for abstract TC but they seem to do unreasonable harm to IQx_KT quants rather than help them or leave them mostly unchanged, which is why I suspect there must be some fine tuning on some level.

Maybe it's the "slop" factors added to dequantize_block_iq2_kt and dequantize_block_iq3_kt and dequantize_block_iq4_kt?

`    const float dl = scale * iq4k_values[((x[i].scales[(ib/4)%4] >> 4*(ib/16)) & 0xf)] * 31.75f * 1.05f;
`
`    const float dl = scale * ((x[i].scales[(ib/4)%4] >> 4*(ib/16)) & 0xf) * 31.75f * 1.01f; //1.015f;
`
`    float scale = dptr[0] * 31.75f * 1.01f;`

Are the 5%, 1%, and 1% just something added to avoid overflow or to use the distribution slightly more optimally? Should they be changed if I adjust the multiplier in 3INST? What else (if anything) would need to change?

[ BTW there seem to be some small inconsistencies between convert.cu and iqk_gemm_ktquants.cpp where the former uses 5%, 1%, 1% and the latter still uses 5%, 1.5%, 1%. ]

Also, if you want KT quants to run even faster, the QTIP paper mentions how to combine the 2 masks in 3INST (AND + XOR) into a single LOP3 instruction. It needs to be added in asm because nvcc can't find this optimization but it improves speed by a measurable amount.

```
    val = ka*val + kb;
    s = (val & kmask) ^ km32;
```
would become something like this (with slightly different asm input params if you want to use your current variable names)
```
        x *= 89226354u;
        x += 64248484u;
        asm volatile ("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x));
```

---

👤 **ikawrakow** commented the **2025-06-02** at **05:26:12**:<br>

> Could current KT code paths be implicitly tuned to expect certain behavior the default parameters provide? I haven't gone through the code super carefully but at first glance I can't immediately figure this out.

The quantization implementation does not attempt to find the provably optimum solution to the RMSE minimization problem for 2 reasons:
* I'm not a GPU person, so prefer to work on the CPU. Solving exactly on the CPU is simply prohibitive.
* All my past experience tells me that a lower RMSE does not necessarily translate into a better observable model quality

Hence, a heuristics is used to determine "optimum" quants. The heuristics is tuned to the specific values being produced by the trellis. But I don't expect you to observe "unreasonable harm", just perhaps a somewhat lower quantization.

I did play quite a bit with different generators when working on #113. For instance, I experimented with using the sum of the 8 bytes of 64-bit random variables. This has many advantages to the QTIP trellises:
* It produces a much better Gaussian distribution, so it is "theoretically better"
* It is much cheaper to generate. There are high quality pseudo random number generators that only require cheap xors and shifts instead of extremely expensive 32-bit integer multiplications. Summing up the elements is fast on CUDA and on the CPU.
* We end up with 16-bit integer random variables, so computing dot products is nearly 2X the speed of the QTIP trellises when there is no native `fp16` support as it is the case on many CPUs. We could go even a step further and squeeze them to 8-bit, which will make also CUDA run significantly faster.

But despite the "theoretical advantage", I observed lower quality quantization. My guess: model weights are not really Gaussian, the outliers are very important, and the "3INST" trellis somehow fits better to real world model weights.

 Concerning `1.05f, 1.015f` etc.: these are fudge factors. They should have been absorbed into the row scales. The reason they ended up like that is that when I was experimenting, it was much cheaper to change a fudge factor in the CUDA code and recompile, than to change it in the quantization code and re-quantize. The fudge factors provide a fairly minor tuning, and the difference between the inconsistent `IQ3_KT` fudge factors is very small. But thanks for bringing it up.

> Also, if you want KT quants to run even faster, the QTIP paper mentions how to combine the 2 masks in 3INST (AND + XOR) into a single LOP3 instruction. It needs to be added in asm because nvcc can't find this optimization but it improves speed by a measurable amount.

I noticed it too in the QTIP paper, but I did not take it seriously because an integer multiplication is quite a bot slower than a xor. But if you say that you observe a measurable performance difference, I'll try it. Thanks!