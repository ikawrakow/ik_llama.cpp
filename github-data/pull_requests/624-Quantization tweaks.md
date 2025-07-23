### 🔀 [#624](https://github.com/ikawrakow/ik_llama.cpp/pull/624) - Quantization tweaks

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-07-17 |
| **Updated** | 2025-07-19 |

---

#### Description

Minor tweaks in the quantization methods for `Q2_K, Q3_K, Q4_K, Q5_K, IQ2_KS, IQ3_KS, IQ3_K`.

Also changed the automatic recipes to use `IQ2_KL` instead of `Q2_K`.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-07-17** at **16:32:39**:<br>

> You devised small gains on perplexity for all those ggml_types, I presume, besides the works on the ftypes/quant strategies?

Yes. But it is basically the same trick.

Most of the heavy-duty lifting during quantization is in determining the block scales. The block scales are floats and then get rounded to an integer in a way depending on how many bits we are spending for block scales. Typically this is just round-to-nearest from a super-block or tensor row scale. While working on `IQ2_KL` I decided to see what happens if I also check the nearest integer values for a block scale, and pick the integer value that minimizes RMSE (changing the block scales can change the quant values, which can sometimes result in a lower difference to the original model weights). This did give a small but non-negligible improvement for `IQ2_KL`. So, today I decided to see if the same trick can be applied  to other quantization types, and the PR includes changes to those types where it helped.

But as perplexity does not tell us anything, I did not post any PPL changes.

Just kidding. I felt lazy to do the usual evaluation with multiple models, so that's why I'm not posting PPL results. I expect people to try and will tell me if it became better.  But it is not a major improvement, just a relatively minor tweak.

---

👤 **ikawrakow** commented the **2025-07-18** at **05:05:47**:<br>

@ubergarm

Thank you for this plot. So, the pure `IQ1_KT` model is basically on par with Unsloth's `IQ1_S`, while being 22% smaller! 

Isn't the bpw for "badname-UD-TQ1_0" wrong? This model shows as just 245 GB on HF (or is HF also wrong about model sizes now?). 

I see `UD-IQ1_S` labeled as "nofmoe". Does this mean that `-fmoe` is not working? I saw elsewhere a report about models failing with `-fmoe`, but no-one would bother to post the model quant composition so I can try to understand what is wrong. If `UD-IQ1_S` is failing with `-fmoe`, can you open an issue for that? Thanks.

---

👤 **ikawrakow** commented the **2025-07-18** at **06:58:19**:<br>

> The IQ2_KS looks slightly better, but the IQ3_KS seemed worse for this PR. Haven't tried others or any other tests.

This is strange. Because of the worse result for `IQ3_KS` for Kimi-2, I now ran perplexity calculations for my usual set of 5 models: LlaMA-1-7B, LlaMA-2-7B, Mistral-7B<sup>1</sup>, LlaMA-3.1-Instruct-8B, DeepSeek-Lite, and also added Qwen3-22B-A3B. Here are the PPL results for Wikitext2 for 2 different context lengths using (almost) pure `IQ3_KS` quantization (only `attn_v` is `IQ4_KS`, token embeddings and output are left at `Q8_0` to not have irrelevant effects from these two tensors)

| Model | Context | PPL (main) | PPL (PR) |
| ---: | ---: | ---: | ---: |
| LlaMA-1-7B |    512 | 6.1930 | 6.1807 |
|                      | 2048 | 5.3355 | 5.3211 |
| LlaMA-2-7B |    512 | 6.1114 | 6.1001 |
|                       | 2048 | 5.3355 | 5.3211 |
| Mistral-7B.   |   512 | 5.9519 | 5.9330 |
|                       | 2048 | 5.0769 | 5.0603 |
| LlaMA-3-8B |  512 | 8.1346 | 8.1198 |
|                       | 2048 | 7.0888 | 7.0715 |
| DeepSeek    |  512 | 7.0893 | 7.0834 |
|                      | 2048 | 6.2253 | 6.2164 |
| Qwen3         | 512 | 9.5122 | 9.4694 |
|                     | 2048 | 8.1964 | 8.1604 |

We see a small but consistent improvement for all 12 cases.

How was the imatrix for Kimi-2 generated? 

___
<sup>1</sup> Why use such ancient models? The LLaMA-v1 models were the basis for k-quants development. i-quants were developed using LLaMA-v1, LLaMA-v2 and Mistral-7B. In my experience, if a quantization technique does well on all 3 of these, it is (almost) guaranteed to do well on any other model out there.

---

👤 **ubergarm** commented the **2025-07-19** at **15:08:07**:<br>

@ikawrakow 

* [ubergarm-imatrix-calibration-corpus-v02.txt](https://gist.github.com/ubergarm/edfeb3ff9c6ec8b49e88cdf627b0711a)
* [Qwen3-14B imatrix dat with above corpus](https://huggingface.co/ubergarm/Qwen3-14B-GGUF/blob/main/imatrix-v02-Qwen3-14B-BF16.dat)
* [Kimi-K2-Instruct imatrix dat with above corpus](https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF/blob/main/imatrix-Kimi-K2-Instruct-Q8_0.dat)

I'd like to spend some time improving my automation/scripts to remove the human error in making these graphs at some point. Thanks for rolling with what we have so far!