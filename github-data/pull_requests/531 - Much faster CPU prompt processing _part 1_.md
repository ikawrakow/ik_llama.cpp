### 🔀 [#531](https://github.com/ikawrakow/ik_llama.cpp/pull/531) - Much faster CPU prompt processing (part 1)

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-16 |
| **Updated** | 2025-06-17 |

---

#### Description

This PR is a continuation of #515, #516, #517, #518 with the following differences
* Quants are repacked to `Q8_K_R8` instead of `Q8_0_R8`. `Q8_K_R8` is the fastest quant known to human kind (see #141), and that helps achieve significant performance gains when batch size is greater than 32 tokens or so
* The technique of on-the-fly repacking before matrix multiplications is extended to a larger set of quants: `IQ1_M, IQ2_XS, IQ2_S, Q3_K` in addition to `IQ1_S, IQ2_XXS, IQ3_XXS, IQ3_S` already improved in the quoted PRs 
* There is also `Q6_K` added, but in this case repacking is to `Q8_0_R8` as `Q6_K` cannot be losslessly repacked to `Q8_K`, and I was worried that there could be a non-negligible accuracy loss due to that. 

The following table shows a PP-512 performance comparison between the main branch and this PR. Model is LlaMA-3.1-8B-Instruct. Quantization is always "pure" (i.e., all tensors except the output tensor and the token embedding tensor are quantized with the selected quantization type). CPU is Ryzen-7950X

| model            |       size |          test |              t/s |              t/s |  Speedup |
| -----------------| ---------: | ------------: | ---------------: | ---------------: | -------: |
| llama 8B IQ1_S   |   2.07 GiB |         pp512 |    264.36 ± 0.32 |    308.67 ± 3.45 |  1.168   |   
| llama 8B IQ1_M   |   2.21 GiB |         pp512 |     25.12 ± 0.15 |    309.81 ± 2.78 | 12.333   |   
| llama 8B IQ2_XXS |   2.35 GiB |         pp512 |    284.22 ± 2.46 |    344.02 ± 4.27 |  1.210   |   
| llama 8B IQ2_XS  |   2.56 GiB |         pp512 |    108.77 ± 2.32 |    346.11 ± 2.26 |  3.182   |   
| llama 8B IQ2_S   |   2.76 GiB |         pp512 |    101.43 ± 1.13 |    341.02 ± 1.60 |  3.362   |   
| llama 8B IQ3_XXS |   3.17 GiB |         pp512 |    280.56 ± 3.15 |    341.95 ± 3.33 |  1.219   |   
| llama 8B Q3_K    |   3.41 GiB |         pp512 |    178.56 ± 2.99 |    344.45 ± 4.15 |  1.929   |   
| llama 8B IQ3_S   |   3.47 GiB |         pp512 |    283.86 ± 2.62 |    340.68 ± 2.87 |  1.200   |   
| llama 8B Q6_K    |   6.14 GiB |         pp512 |    178.49 ± 1.78 |    271.50 ± 2.96 |  1.521   |   

A few notes:
* Gains for the quants that already had repacking to `Q8_0_R8` (`IQ1_S, IQ2_XXS, IQ3_XXS, IQ3_S`) are in the range of 15-20%
* `IQ1_M` stands out because it did not have a fast `iqk` GEMM implementation at all, so we gain a factor of 12X!
* The PR changes the status of i-quants from being slow for CPU inference to being among the fastest (well, at least at this point before I apply this technique to `IQX_K` quants).

I have the impression that most people use `ik_llama.cpp` for MoE models. MoE models are quite different compared to dense models such as LLaMA-3.1-8B because each routed expert "sees" a small fraction of the tokens in a batch, so effective batch size is much smaller compared to a dense model. Hence, PP performance gains for MoE models will be more modest. It is instructive to look as PP performance as a function of batch size. The following graph shows the result for `Q3_K`, which has a reasonably efficient `iqk` GEMM implementation. The repacking strategy kicks in at 32 tokens, so up to that point performance is the same. The relative performance gain from this PR then slowly grows to about 1.9X at 256 tokens, and remains (nearly) the same from there on.  

![z2](https://github.com/user-attachments/assets/34c92f90-ff68-427d-8232-720bcaddec30)

Based on this we can expect lower performance gains for a MoE model. For instance, DeepSeek-R1/V3 have 256 total experts but only 8 active experts, so effectively this strategy will not become active (or will have a very small impact) up to u-batch sizes of 1024 tokens. I cannot run DeepSeek-R1/V3, but I can run Qwen3-30B-A3B, and the next graphs shows performance for this model quantized with `Q3_K`. As expected, performance gains are smaller, about 1.4X at the peak, and poerformance improvement is not significant before 64 tokens.

  
![z3](https://github.com/user-attachments/assets/6370ace4-3ae6-4e3e-a5d0-a5846f4ed63a)

---

#### 💬 Conversation

👤 **saood06** commented the **2025-06-16** at **10:26:55**:<br>

Does this also improve the behavior at higher contexts? For me running Deepseek at higher contexts PP and TG both approach ~1 t/s at high context.

---

👤 **ikawrakow** commented the **2025-06-16** at **10:31:53**:<br>

> For me running Deepseek at higher contexts PP and TG both approach ~1 t/s.

This indicates that your computer spends the entire time computing self attention for long enough context. If so, this PR will have zero impact on your long context performance.

---

👤 **ikawrakow** commented the **2025-06-16** at **12:53:47**:<br>

> but at higher context the power usage looks a lot closer to TG (which is memory/QPI bandwidth bound).

Or is it rather the other way around (TG looks a lot closer to PP)? If you buy my explanation that for a large context all the time is spent in the self attention calculation, then there isn't that much of a difference between TG and PP: for DeepSeek each row in the KV  cache multiples 128 rows of activations (`K*Q` and `V*softmax(K*Q)`), so the matrix multiplications in TG and PP have very similar characteristics (there isn't much of a difference between multiplying 128 rows and 128 x n_ubatch rows), and it is compute bound, not memory bound.

---

👤 **saood06** commented the **2025-06-16** at **13:54:42**:<br>

>If you buy my explanation

I do, I was just trying to understand it.

> Or is it rather the other way around (TG looks a lot closer to PP)? that for a large context all the time is spent in the self attention calculation, then there isn't that much of a difference between TG and PP: for DeepSeek each row in the KV cache multiples 128 rows of activations (`K*Q` and `V*softmax(K*Q)`), so the matrix multiplications in TG and PP have very similar characteristics (there isn't much of a difference between multiplying 128 rows and 128 x n_ubatch rows), and it is compute bound, not memory bound.

That makes sense. 

I did attempt to look at the [PCM](https://github.com/intel/pcm) data I had from earlier and just generated, and looked at CPU power usage and IPC but I'm not sure if the numbers are actually useful since I found during TG that it was causing paging (there really isn't much spare RAM on my system during inference).

---

👤 **ubergarm** commented the **2025-06-16** at **23:06:48**:<br>

Not a comprehensive test, but this `PR531` does indeed speed-up PP as
compared to `main` on my DeepSeek-R1-0528-IQ1_S.

So while not as dramatic given only 58 `ffn_down_exps@iq1_m` on this MoE,
the `iq1_s` speed-ups are already merged into main so overall much faster
than before.

The `IQ1_S_R4` still benches faster for this specific configuration at least.

Note, to keep it simple, I did *not* use `-rtr` to repack the attn/shexp
tensors; so actual CPU-only scenario would likely be faster still.

## DeepSeek-R1-0528-IQ1_S
- type    f32:  361 tensors
- type   q4_0:   61 tensors `attn_k_b`
- type  iq1_s:  116 tensors `ffn_(gate|up)_exps`
- type  iq1_m:   58 tensors `ffn_down_exps`
- type iq4_ks:  551 tensors `everything else`

## DeepSeek-R1-0528-IQ1_S_R4
- type      f32:  361 tensors
- type     q4_0:   61 tensors `attn_k_b`
- type iq1_s_r4:  116 tensors `ffn_(gate|up)_exps`
- type iq1_m_r4:   58 tensors `ffn_down_exps`
- type   iq4_ks:  551 tensors `everything else`

Importantly, `llama-perplexity` runs clean on PR531@72fd9faa so the new `iq1_m` implementation seems solid.

* `IQ1_S`: `Final estimate: PPL = 4.8910 +/- 0.02856`
* `IQ1_S_R4`: `Final estimate: PPL = 4.8805 +/- 0.02876` (computed back on PR494)

![sweep-bench-PR31](https://github.com/user-attachments/assets/98b1266a-cbfe-4794-950d-9bee98983280)

---

👤 **ikawrakow** commented the **2025-06-17** at **10:32:11**:<br>

> The IQ1_S_R4 still benches faster for this specific configuration at least and seems to be the same speed on both this PR and main as I would expect.

This is because of the extremely high total_experts/active_experts=32 ratio in DeeSeek-V3. For u_batch size of 512 we are still far away from the regime where this new repacking scheme pays large dividends. Perhaps the gains will be bigger for `u_batch = 1024` or even `u_batch = 2048`?

But yes, I see that this PR may not have the huge impact that it should because people have somehow decided that `ik_llama.cpp` is only good for very large MoE models, so they keep using `llama.cpp` for everything else, missing out big times on performance for CPU-only inference (and it isn't so that CPU performance is not discussed in the `llama.cpp` repository on a regular basis).

---

👤 **saood06** commented the **2025-06-17** at **20:56:40**:<br>

>For me running Deepseek at higher contexts PP and TG both approach ~1 t/s.

I had been so used to V3 where I never enabled high batch sizes with amb because I rarely requested over the default batch size of 512. But with R1 that is not in the case (due to thought tokens removal which results in reprocessing context).

I ran an experiment at high context, processing 4096 tokens (33640 to 37736) and this went from 2950 to 1619 seconds, and even a reduction in compute buffer (`15387.76 MiB` vs `9404.80 MiB`).