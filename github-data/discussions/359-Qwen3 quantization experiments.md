### 🗣️ [#359](https://github.com/ikawrakow/ik_llama.cpp/discussions/359) - Qwen3 quantization experiments

| **Author** | `ikawrakow` |
| :--- | :--- |
| **Created** | 2025-04-30 |
| **Updated** | 2025-06-11 |

---

#### Description

I did some experimentation with Qwen3 quantization. As I don't have the horsepower to run the flagship model, I experimented with the Qwen3-30B-A3B MoE model. I'm reporting the results here, hopefully this could be useful also for Qwen3-235B-A22B.

The following graph shows a comparison between the Unsloth so called "dynamic" quants and the quantization mixes I prepared. The Unsloth quantized models, shown with black symbols, are from [their HF repository](https://huggingface.co/unsloth/Qwen3-30B-A3B-128K-GGUF), and the text in black besides the data points gives the corresponding file name. The red symbols are for my quantization mixes and their recipes will be given below. The x-axis is model size in GiB (and not GB, as HF likes to use). The y-axis is the quantization error in percent defined as `PPL(Q)/PPL(bf16)-1`. Based on these results, it does not look like Unsloth did a particularly good job with their "dynamic" quants for this model. One can get the same quantization quality with ~2 GiB smaller model, so nearly 20% smaller at the low-bpw end.
   
![q3](https://github.com/user-attachments/assets/700922a6-9dc9-40ce-9080-7120887ae801)

My recipes are almost entirely composed of `IQK` quants, so exclusive to this repository. I did not go beyond 4.3 bpw as there the quantization error is 0.57% (and I have seen sub-1% quantization error to be called "lossless" in the quantization literature).

### Recipe IQK-1

```
./bin/llama-quantize --imatrix $imatrix \
            --custom-q "attn=iq5_k,token_embd.weight=q4_K,output.weight=q6_K"
            --custom-q "blk\.[0-5]\.ffn_down_exps=iq4_ks,ffn_down_exps=iq2_ks"
            --custom-q "exps=iq2_ks" .\
            Qwen3-128x1.8B-BF16.gguf $model_file_name iq2_ks
```
Note that one can combine all arguments following `--custom-q` into a single, comma separated list of regular expressions. I have split into several `--custom-q` arguments for better readability.  So, basically, all attention tensors quantized with `IQ5_K`, the first 6 layers of `ffn_down_exps` with `IQ4_KS`, everything else with `IQ2_KS`. Oh, here and for all other recipes, token embeddings are `Q4_K` and the output tensor is `Q6_K`. This quantized model ends up being 8.745 GiB, so only very slightly larger than Unsloth's `UD-IQ1_S` (8.396 GiB). 

### Recipe IQK-2

```
./bin/llama-quantize --imatrix $imatrix \
        --custom-q "attn=iq5_k,token_embd.weight=q4_K,output.weight=q6_K" \
        --custom-q "blk\.[0-5]\.ffn_down_exps=iq4_ks,ffn_down_exps=iq2_k,exps=iq2_k" \
        Qwen3-128x1.8B-BF16.gguf $model_file_name iq2_k
```
Very similar to Recipe-1, with all attention tensors quantized with `IQ5_K`, the first 6 layers of `ffn_down_exps` with `IQ4_KS`, all other experts with `IQ2_K`. The quantized model ends up being 9.314 GiB.

### Recipe IQK-3

```
./bin/llama-quantize --imatrix $imatrix \
        --custom-q "attn=iq5_k,token_embd.weight=q4_K,output.weight=q6_K" \
        --custom-q "blk\.[0-5]\.ffn_down_exps=iq4_k,ffn_down_exps=iq3_k,exps=iq2_k"  \
         Qwen3-128x1.8B-BF16.gguf $model_file_name iq2_k
```
The difference to Recipe IQK-2 is that the first 6 layers of `ffn_down_exps` is quantized with `IQ4_K`, the remaining `ffn_down_exps` tensors with `IQ3_K`. The quantized model size is 10.389 GiB.

### Recipe IQK-4

```
./bin/llama-quantize --imatrix $imatrix \
        --custom-q "attn=iq5_k,token_embd.weight=q4_K,output.weight=q6_K" \
        --custom-q "blk\.[0-5]\.ffn_down_exps=iq4_k,ffn_down_exps=iq3_k" \
        --custom-q "blk\.[0-9]\.ffn=iq3_k,blk\.1[0-5]\.ffn=iq3_k,blk\.4[0-9]\.ffn=iq3_k" \
        --custom-q "exps=iq2_k" Qwen3-128x1.8B-BF16.gguf $model_file_name iq3_k
```
Similar to Recipe IQK-3, but now the first 16 and the last 8 layers of the `ffn_up_exps` and `ffn_gate_exps` tensors are quantized with `IQ3_K`. The quantized model size is 11.584 GiB.

### Recipe IQK-5

```
./bin/llama-quantize --imatrix $imatrix \
        --custom-q "attn=iq5_k,token_embd.weight=q4_K,output.weight=q6_K" \
        --custom-q "blk\.[0-5]\.ffn_down_exps=iq4_k,ffn_down_exps=iq3_k,exps=iq3_k" \
         Qwen3-128x1.8B-BF16.gguf $model_file_name iq3_k
```
I.e., all experts are `IQ3_K`, except for the first 6 layers of `ffn_down_exps`, which are `IQ4_K`. Model size is 12.779 GiB

### Recipe IQK-6

```
./bin/llama-quantize --imatrix $imatrix \
        --custom-q "attn=iq5_k,token_embd.weight=q4_K,output.weight=q6_K" \
        --custom-q ".*=iq4_ks" Qwen3-128x1.8B-BF16.gguf $model_file_name" iq4_ks
```
I.e., all tensors (except attention, output and embeddings) are `IQ4_KS`. The quantized model size is 15.454 GiB.

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2025-04-30** at **09:11:48**:<br>

Has there been any QAT going on with the Qwen3 models? I didn't see anything mentioned in the [linked blog post](https://qwenlm.github.io/blog/qwen3/), but there are indications that QAT may have been involved. Does somebody know?

> 👤 **saood06** replied the **2025-04-30** at **09:32:22**:<br>
> >but there are indications that QAT may have been involved.
> 
> What indications are you referring to?
> 
> 👤 **ikawrakow** replied the **2025-04-30** at **09:34:07**:<br>
> I'm putting together the results and will post in a bit. In the meantime I was curious if somebody knew if QAT was used for Qwen3.

---

👤 **ikawrakow** replied the **2025-04-30** at **12:25:59**:<br>

# QAT used in Qwen3 training?

After posting the above results, I decided to see what I get with `IQ4_K` quantization. `IQ4_KS`, which is 4.25 bpw,  had arrived at a quantization error of 0.6%. `IQ4_K` is 4.5 bpw and normally better than `IQ4_KS`, and I was thinking that it may get into 6-bit territory. It uses blocks of 16 vs blocks of 32 for `IQ4_KS`. Other than that, the two quants are extremely similar (same non-uniform grid, same quantization approach). To my surprise, `IQ4_K` arrived at a slightly higher perplexity than `IQ4_KS`. So, I thought "OK, there is something funny going on here. What if I replaced `IQ4_K` with `IQ4_KS` in the above quantization mixes, and for good measure also used `IQ4_KS` instead of `IQ5_K` for the attention tensors?". I started with the Recipe IQK-1, and PPL dropped from 10.09 to 9.95, while decreasing the model size from 8.745 GiB to 8.615 GiB (not a big reduction, but for very low bpw quants quantization error increases quite fast with decreasing model size, so reducing model size and PPL is quite of an effect).

OK, then, let's just redo all recipes, using `IQ4_KS` instead of `IQ5_K` or `IQ4_K`.  The Wiki2 perplexity for the `bf16` model is `9.0656`, and here are the new results for the 6 recipes:

| Recipe | Model size | PPL |
| ---: | ---: | ---: |
| IQK-1 | 8.615 GiB | 9.9517 |
| IQK-2 | 9.183 GiB | 9.7154 |
| IQK-3 | 10.229 GiB | 9.3908 |
| IQK-4 | 11.454 GiB | 9.1057 |
| IQK-5 | 12.620 GiB | 9.0147 |
| IQK-6 | 15.324 GiB | 8.9873 | 

Oops. Recipes 5 and 6 have a lower PPL than the `bf16` model!

Hmm, my `bf16` value must be wrong. Let's recompute that with mainline. And to not take any chances, let's not use `Q8_0` as a surrogate for `bf16`. Which, given my 16 GB GPU, basically means computing on the CPU. Mainline is slow as molasses on the CPU, but still let's see. 50 minutes later: mainline `PPL = 9.0665`. 

Oops.

OK, it must be my imatrix. People have filled whole libraries writing about how the imatrix calibration data needs to be random, diverse, whatnot. OK, let's grab the [Unsloth imatrix](https://huggingface.co/unsloth/Qwen3-30B-A3B-128K-GGUF/blob/main/imatrix_unsloth.dat). Quantize, run `llama-perplexity` for recipe IQK-6 . Result: `PPL = 8.8787`. 

Oops. That's definitely even less diverse than mine.

Let's grab [Bartowski imatrix](https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF/blob/main/Qwen_Qwen3-30B-A3B.imatrix). Quantize recipe IQK-6, run `llama-perplexity`. Result: `PPL = 8.9727`.

Oops. It looks like mine, obtained 100% from `wiki.train.raw`, is more diverse than theirs (as it over-fits `wiki.test.raw` less). I did use way more batches than they did, but still.

What happens if I don't use an imatrix at all? Quantize recipe `IQK-6`, run `llama-perplexity`. Result: `PPL = 9.3119`.

So, that's about 2.6% quantization error, so in the range of what I would expect from `IQ4_KS` without imatrix. Oh, wait.  I happen to know that when no imatrix is provided, the quantization function uses $w_i = x_i^2$ as the importance of model weight $x_i$. This gives more importance to larger magnitude weights. Historically, from the pre-matrix days, this was the best strategy and always resulted in a better quantization than just assigning the same importance to all model weights. But if QAT was involved in the training, and if the model weights have been forced (guided) against some `fp4` variant (more on this below), then doing that will be detrimental to quantization accuracy. So, let's just set `w_i = 1`, quantize again, run `llama-perplexity`. Result: `PPL = 9.1470`. That's just 0.9% higher than `bf16` using 4.25 bpw. The table below summarizes the above observations:

| What | PPL |
| ---: | ---: |
| bf16 model | 9.0656 |
| No imatrix, w = x^2 | 9.3119 |
| No imatrix, w = 1 | 9.1470 | 
| IK imatrix | 8.9873 |
| Bartowski imatrix | 8.9727 |
| Unsloth imatrix | 8.8787 |

So, what if they have used some form of QAT targeted towards some `fp4` variant? `fp4` does have just 16 distinct values, and, having a mantissa and an exponent, possible values are non-uniformly distributed between a min and a max value. This is kind of similar to the non-linear quantization types `IQ4_NL, IQ4_XS, IQ4_KS, IQ4_K`, so let's take look. The following graph compares `nf4` and `fe1m2` to the `iq4k_values` used for `IQ4_K` and `IQ4_KS`. The thick black line illustrates linear mapping. The four different `iq4k` variants are all achievable, depending on the sign of the block scale and the block shift bit (the sign of the block scale does not matter for the `fp4` values as they are symmetric).

![nf4](https://github.com/user-attachments/assets/660843a8-f56c-4c76-a0a0-da9d4cb3f21c)

Looking at this graph, it seems plausible that if `fp4` QAT was used with blocks of 32, `IQ4_KS` would adapt quite well to that. 

Just in case, I also checked PPL for `Q4_0`. Without imatrix we get `PPL = 9.3017`, so no, unlike Google with Gemma3, the Qwen3 creators have not been overfitting to `Q4_0`.

> 👤 **saood06** replied the **2025-04-30** at **23:41:35**:<br>
> Very interesting will definitely take this into account when making my own mixes of Qwen-3.
> 
> > OK, it must be my imatrix. People have filled whole libraries writing about how the imatrix calibration data needs to be random, diverse, whatnot. OK, let's grab the [Unsloth imatrix](https://huggingface.co/unsloth/Qwen3-30B-A3B-128K-GGUF/blob/main/imatrix_unsloth.dat). Quantize, run `llama-perplexity` for recipe IQK-6 . Result: `PPL = 8.8787`.
> > 
> > Oops. That's definitely even less diverse than mine.
> > 
> > Let's grab [Bartowski imatrix](https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF/blob/main/Qwen_Qwen3-30B-A3B.imatrix). Quantize recipe IQK-6, run `llama-perplexity`. Result: `PPL = 8.9727`.
> > 
> > Oops. It looks like mine, obtained 100% from `wiki.train.raw`, is more diverse than theirs (as it over-fits `wiki.test.raw` less). I did use way more batches than they did, but still.
> 
> Are you sure about the other two imatrix overfitting? Do you have any data showing they perform worse when testing things other than `wiki.test.raw`?
> 
> Also on a somewhat related note, I know you said elsewhere "I have written elsewhere about the equivalence of PPL and KLD for an infinitely large test corpus, and about the superiority of PPL for a test corpus of limited size, so I will not repeat myself here." so sorry if this is a repeat but have you heard the argument in [this paper](https://arxiv.org/abs/2407.09141) which has this section critiquing PPL.
> 
> >Though we have focused on accuracy so far, our observation that the difference between two models’ output token values cancel out leaving the average metric result unchanged, is applicable to perplexity as well. In particular, since perplexity may be interpreted as the inverse of the geometric mean of token probabilities, lower probabilities for some tokens in the test dataset may be cancelled by higher probabilities of other tokens. This indicates that perplexity alone is also inadequate in evaluating model compression schemes. Therefore, we argue that along with perplexity, KL-Divergence between the distributions generated by the baseline and optimized models should also be reported.
> >
> >Figure 9 in Appendix plots the log-likelihood difference between the 16-bit and quantized model for each of the tokens in the wiki-2 dataset Merity et al. (2016) for four different quantization schemes. From the figure, it appears that the log-likelihoods of the quantized model is just the log-likelihood of baseline model with some symmetric noise added. Now, since perplexity is e−avg(logprobabilities), adding any amount of symmetric noise leaves it unchanged. For example, addition of Gaussian noise to the log-probability outputs of the model should maintain the perplexity, while the quality of generation will degrade as the standard deviation of the noise increases (see Table 19). This analysis demonstrates one key weakness with the perplexity metric when used for evaluating compression techniques. While it is not clear if adding Gaussian noise to the log-likelihoods is an accurate representation of the behavior of compression schemes, it appears to be a good analogy. As we shall see in Section 6, as quantization increases, there is steady degradation in the quality of the text generated by the model that are visible only by examining them closely.
> 
> 👤 **ikawrakow** replied the **2025-05-01** at **06:15:01**:<br>
> They critique PPL. Do you want me to critique the paper for you?
> 
> 👤 **saood06** replied the **2025-05-01** at **06:57:34**:<br>
> > They critique PPL. Do you want me to critique the paper for you?
> 
> I'm not asking for a critique and I don't really care for the paper as they heavily imply there is an objective measure of performance of an LLM, but in my view there isn't one and it is all dependent on one's use case and use of the LLM (prompting, sampling, etc.), it's just they state, "While it is not clear if adding Gaussian noise to the log-likelihoods is an accurate representation of the behavior of compression schemes, it appears to be a good analogy. ", and I don't have any intuition of whether or not that statement is correct, but thought you might have a take on that if you don't mind sharing.
> 
> 👤 **ikawrakow** replied the **2025-05-01** at **18:02:40**:<br>
> >  it's just they state, "While it is not clear if adding Gaussian noise to the log-likelihoods is an accurate representation of the behavior of compression schemes, it appears to be a good analogy. ", and I don't have any intuition of whether or not that statement is correct, but thought you might have a take on that if you don't mind sharing.
> 
> This is indeed one critique of their argument, and they deal with it in a very hand wavy way. Apart from the overall quality of the paper, if I just focus on their Table 19, which is the crux of their argument against PPL, here are several other points:
> * Why is it that the table does not contain the response of the model without Gaussian noise added? If I'm making the argument that Gaussian noise degrades quality without changing PPL, then I need to show this to be true. I do that by asking the exact same question for each noise level, or if I for some reason decided to go the strange route of giving the LLM a task with increasing difficulty, then I definitely need to also show the response without noise
> * Did they actually compute PPL? The odds that it remained exactly the same with 5 standard deviations of noise added are pretty much zero.
> * What did they do with KLD? Did they add Gaussian noise just to the top token? Or did they add noise to all tokens? They don't say. I happen to expect that if they added Gaussian noise to all predicted probabilities, they would observe a statistically insignificant change in KLD
> * Let's look at the amount of noise added:
>   - Is it added to the probabilities `p` or to `ln(p)`? They don't say
>   - What does `0.0, 1.0, 2.0, ..., 5.0` mean? Is this measured in standard deviations, or is it the actual width of the Gaussian? If the former, they should have specified the model so we can see the standard deviation in Fig 9. But I would guess it is the latter. Do you know what it means to add a Gaussian noise with $\sigma = 5$ to (I assume) the logits? It basically wipes out whatever the model has predicted. And then I wonder what they did in practice. The logits need to be in $\[-\infty, 0)$ (but the important once are concentrated in, say, [-5, 0)). When I add Gaussian noise with $\sigma = 5$, chances are pretty high the logit may go out of the allowed range. What is it that I do then? I clamp it? If I do that, I no longer have the argument that PPL remains unchanged because it doesn't (the noise distribution used is no longer symmetric, so adding noise does modify the expectation value, which is the PPL). 
>   - We do have figure 9 in the paper. There I see that the standard deviation of the difference between the base model and the quantized models changes from small to about 0.2 when I go from 8 bits to 4 bits. Why on earth would I be making experiments with a Gaussian noise of 1,2,3,4, and 5? I have a hunch here. I expect that if they added a Gaussian noise corresponding to the difference between 8-bit and 4-bit quantization, they wouldn't be able to measure the difference. Which would make the entire argument fall apart.
> 
> [Here](https://huggingface.co/blog/bartowski/llama4-scout-off#67f7beac7500c1c63d048419) are graphs that shows KLD vs PPL and correct top token probability vs PPL for the models studied in the blog post. The correlation coefficient for the straight line fits are 99% and 98%, respectively. I'm a physicist, and as part of my physics education I studied statistics. Physics experiments require a lot of effort, so they thought us that it is important to understand that it does not make sense to measure quantities that are highly correlated. When correlation is as high as 98-99%, measuring one lets you predict the other. This is how it is with PPL and KLD, and with PPL and correct top token.
> 
> But if you still have doubts, open a discussion and let's discuss it there. This discussion is about Qwen3 quantization.
> 
> 👤 **bartowski1182** replied the **2025-05-02** at **02:05:54**:<br>
> so strange to see decreasing PPL when quantizing 🤔 
> 
> I suppose one theory could be that by quantizing reduces some of the noise that's correlated with thinking or other stranger text, and so it's more likely to produce wiki text style generation? that wouldn't be absurd
> 
> <details>
> <summary>KLD vs PPL offtopic yapping</summary>
> it does always interest me when PPL and KLD are not *directly* correlated, like how PPL for one quant can decrease faster than the KLD, I completely accept your conclusion that they are correlated, it's quite obvious on many observations you've posted, but does make me curious when they diverge a little
> </details>
> 
> your results do make me wonder about the possibility of QAT training.. feels like they would have spoken about that
> 
> also what do you think about QAT in terms of quantization target?
> 
> Would a QAT for int4 also help Q4_K with its scaling factors? and nf4 with its different format? or would it need to be specifically the same target quant format?
> 
> just thinking out loud
> 
> 👤 **ubergarm** replied the **2025-05-02** at **04:23:54**:<br>
> I don't find any references to QAT for this Qwen3 release either, but the paper itself is not yet linked. I did find some official recommendations on quantizing by the Qwen team including GGUF format, some of the documentation maybe is recycled from previous Qwen2.5 release: https://github.com/QwenLM/Qwen3/tree/main/docs/source/quantization
> 
> 👤 **ikawrakow** replied the **2025-05-02** at **06:18:07**:<br>
> @saood06 
> 
> > Are you sure about the other two imatrix overfitting? Do you have any data showing they perform worse when testing things other than wiki.test.raw?
> 
> It is hard to prove one model is working better than another with just subjective feelings about the quality of the responses. But if we assume that QAT was not involved in the training, and we observe that the quantized model arrives at a lower PPL for a given test corpus than the `bf16` model, than this must be due to overfitting to the specific type of test data. The only way the overfitting can happen is via the imatrix. Hence, one imatrix resulting in a lower PPL than another imatrix can only mean that the first imatrix has been computed with calibration data that is more similar to the test corpus than the calibration data of the second imatrix.
> 
> You see it differently?
> 
> 👤 **bartowski1182** replied the **2025-05-02** at **14:37:59**:<br>
> Also I should note, specifically for the 30B (because I was having issues with experts not being activated) I generated ~100k more tokens of noise from the model which seemed to positively affect the results, there was a bunch of English and Chinese as well as a few other languages I noticed fly by, and a ton of emojis
> 
> But yeah with my usual dataset I couldn't make iq2_xs and smaller from lack of data, after augmenting it I had no issues
> 
> Point being, mine is very likely not overfit 😅

---

👤 **ubergarm** replied the **2025-05-02** at **02:09:33**:<br>

Oh man I just released [ubergarm/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-mix-IQ4_K.gguf](https://www.reddit.com/r/LocalLLaMA/comments/1kcp34g/ubergarmqwen330ba3bgguf_1600_toksec_pp_105_toksec/) just *before* finding and reading this discussion!!! ooops!

I have some PPL data from `wiki.test.raw` as well as my own `ubergarm-kld-test-corpus.txt` and KLD with `ubergarm-kld-test-corpus.txt` using the `bf16` as the baseline. I got `Final estimate: PPL = 9.0703 +/- 0.07223` `wiki.test.raw` on the `bf16` model using `wiki-test.raw` so basically same as yours.

*EDIT*: Add unreleased `ubergarm/IQ4_KS` (pure `iq4_k` all and `q8_0` for non-repeating embedding/output).
*EDIT*: Add unreleased `ubergarm/smol-IQ4_KS` (same as above but `q4_0` tok embedding and `q6_0` out saving ~220KiB)

| What | PPL `wiki.test.raw` | PPL `ubergarm-kld-test-corpus.txt` | Mean KLD | Mean Δp |
| --- | --- | --- | --- | --- |
| bf16 model| 9.0703 +/- 0.07223 | 15.1443 +/- 0.10239 | n/a | n/a |
| Q8_0 | 9.0740 +/- 0.07228 | 15.152095 ± 0.102398 | 0.002337 ± 0.000009 | -0.020 ± 0.003 % |
| ubergarm/mix-IQ4_K | 9.1184 +/- 0.07278 | 15.218819 ± 0.103071 | 0.004821 ± 0.000024 | -0.025 ± 0.004 % |
| ubergarm/IQ4_KS | 8.9862 +/- 0.07061 | 15.182811 ± 0.102278 | 0.014617 ± 0.000068 | -0.209 ± 0.007 % |
| ubergarm/smol-IQ4_KS | 8.9864 +/- 0.07061 | 15.169532 ± 0.102138 | 0.014953 ± 0.000076 | -0.239 ± 0.007 % |
| unsloth/UD-Q4_K_XL | 9.1688 +/- 0.07290 | 15.281833 ± 0.103140 | 0.016495 ± 0.000071 | -0.320 ± 0.008 % |
| bartowski/Q4_K_M | 9.2092 +/- 0.07381 | 15.194468 ± 0.102605 | 0.010136 ± 0.000053 | -0.158 ± 0.006 % |
| bartowski/Q4_K_S | 9.2232 +/- 0.07371 | 15.202408 ± 0.102513 | 0.012915 ± 0.000065 | -0.227 ± 0.007 % |

I have some more KLD and token probability stats too with graphs to make a better write-up eventually.

So sounds like if Qwen was using QAT targeted at fp4, there it may be possible to use IQ4_KS to shave some weight without sacrificing quality? I'll have to try some more mixes...

If I'm following here, sounds like the goal to get as low as possible without going *below* the bf16 PPL? So using `No imatrix, w = 1` would be better than over fitting with a poor imatrix corpus?

> 👤 **saood06** replied the **2025-05-02** at **03:54:37**:<br>
> Thank you for this, it is interesting to see the differences, do you mind doing more tests with the same quant mix but with different imatrix datasets.
> 
> I was experimenting with some things and happened to run a PPL run
> 
> Mix made with: 
> 
> `./bin/llama-quantize --imatrix [...]matrix_unsloth.dat  --custom-q "token_embd.weight=q4_K,output.weight=q6_K" --custom-q ".*=iq4_ks" [...]Qwen3-30B-A3B-128K-BF16-00001-of-00002.gguf iq4_ks`
> 
> PPL run with:
> 
>  `.\llama-perplexity.exe -m "[...]" -ngl 99 -fa -fmoe  -f "[..]wiki.test.raw" -ctk q8_0 -ctv q8_0`
> 
> ```
> [1]5.9247,[2]8.1743,[3]7.8086,[4]7.4665,[5]7.5827,[6]7.9146,[7]7.9256,[8]8.3800,[9]8.7338,[10]9.2358,[11]9.2796,[12]9.4751,[13]9.9094,[14]9.5276,[15]9.4315,[16]9.6651,[17]9.1359,[18]9.2973,[19]9.2420,[20]9.2027,[21]8.8980,[22]8.8728,[23]8.5166,[24]8.0805,[25]7.8595,[26]7.6511,[27]7.4495,[28]7.3339,[29]7.3661,[30]7.3024,[31]7.2536,[32]7.2892,[33]7.1866,[34]7.2554,[35]7.3744,[36]7.4953,[37]7.6588,[38]7.7084,[39]7.7045,[40]7.7682,[41]7.7699,[42]7.7425,[43]7.8004,[44]7.8073,[45]7.8050,[46]7.8210,[47]7.9946,[48]8.0825,[49]8.0659,[50]8.1399,[51]8.1844,[52]8.2167,[53]8.2830,[54]8.3550,[55]8.3549,[56]8.3934,[57]8.3696,[58]8.4045,[59]8.4742,[60]8.5250,[61]8.5489,[62]8.5950,[63]8.6515,[64]8.7124,[65]8.7947,[66]8.8633,[67]8.9378,[68]8.9229,[69]8.9382,[70]8.9302,[71]8.9595,[72]9.0276,[73]9.0641,[74]9.0806,[75]9.0334,[76]9.0309,[77]9.0764,[78]9.1284,[79]9.0602,[80]9.0377,[81]8.9987,[82]9.0382,[83]9.0019,[84]8.9840,[85]9.0000,[86]9.0918,[87]9.1256,[88]9.1180,[89]9.1208,[90]9.0922,[91]9.1438,[92]9.1152,[93]9.1593,[94]9.1677,[95]9.1424,[96]9.1272,[97]9.1001,[98]9.1115,[99]9.0908,[100]9.1484,[101]9.1762,[102]9.1625,[103]9.1782,[104]9.1421,[105]9.1415,[106]9.1300,[107]9.1597,[108]9.1980,[109]9.2245,[110]9.2765,[111]9.3936,[112]9.3875,[113]9.3438,[114]9.4089,[115]9.4185,[116]9.3804,[117]9.3632,[118]9.3410,[119]9.2961,[120]9.3076,[121]9.2886,[122]9.2788,[123]9.2384,[124]9.1841,[125]9.1533,[126]9.1287,[127]9.0717,[128]9.0405,[129]9.0034,[130]8.9708,[131]8.9196,[132]8.8810,[133]8.8661,[134]8.8615,[135]8.8584,[136]8.8559,[137]8.8246,[138]8.7906,[139]8.7961,[140]8.7798,[141]8.7714,[142]8.7767,[143]8.7843,[144]8.8132,[145]8.7829,[146]8.7426,[147]8.7048,[148]8.6669,[149]8.6472,[150]8.6109,[151]8.5820,[152]8.5675,[153]8.5704,[154]8.5259,[155]8.5278,[156]8.4865,[157]8.4598,[158]8.4230,[159]8.3891,[160]8.3530,[161]8.3328,[162]8.3172,[163]8.2974,[164]8.2960,[165]8.2776,[166]8.2697,[167]8.2623,[168]8.2869,[169]8.2879,[170]8.3189,[171]8.3428,[172]8.3957,[173]8.4403,[174]8.4601,[175]8.5162,[176]8.5446,[177]8.5919,[178]8.6328,[179]8.6446,[180]8.6443,[181]8.6652,[182]8.6882,[183]8.6726,[184]8.6863,[185]8.6957,[186]8.6995,[187]8.7099,[188]8.7095,[189]8.7234,[190]8.7530,[191]8.7610,[192]8.7754,[193]8.7698,[194]8.7940,[195]8.8140,[196]8.8248,[197]8.8314,[198]8.8092,[199]8.8022,[200]8.7916,[201]8.7998,[202]8.8119,[203]8.8319,[204]8.8476,[205]8.8637,[206]8.8524,[207]8.8779,[208]8.8596,[209]8.8616,[210]8.8582,[211]8.8598,[212]8.8597,[213]8.8587,[214]8.8400,[215]8.8232,[216]8.8181,[217]8.8283,[218]8.8254,[219]8.7966,[220]8.7662,[221]8.7545,[222]8.7441,[223]8.7394,[224]8.7550,[225]8.7343,[226]8.7351,[227]8.7257,[228]8.7006,[229]8.6707,[230]8.6484,[231]8.6321,[232]8.6174,[233]8.6163,[234]8.6245,[235]8.6190,[236]8.6054,[237]8.5927,[238]8.5727,[239]8.5641,[240]8.5666,[241]8.5683,[242]8.5787,[243]8.5781,[244]8.5955,[245]8.5972,[246]8.6165,[247]8.6211,[248]8.6247,[249]8.6325,[250]8.6405,[251]8.6578,[252]8.6728,[253]8.7001,[254]8.7186,[255]8.7225,[256]8.7368,[257]8.7484,[258]8.7335,[259]8.7119,[260]8.6908,[261]8.6662,[262]8.6495,[263]8.6428,[264]8.6444,[265]8.6552,[266]8.6581,[267]8.6575,[268]8.6485,[269]8.6505,[270]8.6479,[271]8.6384,[272]8.6374,[273]8.6343,[274]8.6304,[275]8.6300,[276]8.6186,[277]8.6173,[278]8.6225,[279]8.6199,[280]8.6151,[281]8.6080,[282]8.6152,[283]8.5876,[284]8.5549,[285]8.5602,[286]8.5455,[287]8.5271,[288]8.5236,[289]8.5263,[290]8.5485,[291]8.5523,[292]8.5515,[293]8.5514,[294]8.5683,[295]8.5875,[296]8.6017,[297]8.6254,[298]8.6212,[299]8.6067,[300]8.6080,[301]8.6052,[302]8.6009,[303]8.5933,[304]8.6112,[305]8.6107,[306]8.6070,[307]8.6090,[308]8.6063,[309]8.6045,[310]8.6092,[311]8.6138,[312]8.6037,[313]8.5963,[314]8.6001,[315]8.5870,[316]8.5918,[317]8.6133,[318]8.6186,[319]8.6134,[320]8.6160,[321]8.6021,[322]8.6136,[323]8.6285,[324]8.6449,[325]8.6652,[326]8.6675,[327]8.6592,[328]8.6594,[329]8.6444,[330]8.6352,[331]8.6262,[332]8.6289,[333]8.6290,[334]8.6211,[335]8.6103,[336]8.6040,[337]8.6102,[338]8.6196,[339]8.6126,[340]8.6043,[341]8.5928,[342]8.5905,[343]8.5851,[344]8.5939,[345]8.5976,[346]8.5932,[347]8.5814,[348]8.5820,[349]8.5750,[350]8.5689,[351]8.5691,[352]8.5742,[353]8.5740,[354]8.5611,[355]8.5775,[356]8.5859,[357]8.5915,[358]8.5794,[359]8.5805,[360]8.5783,[361]8.5832,[362]8.5786,[363]8.5745,[364]8.5847,[365]8.6017,[366]8.6275,[367]8.6443,[368]8.6732,[369]8.6930,[370]8.7077,[371]8.7285,[372]8.7517,[373]8.7593,[374]8.7694,[375]8.7897,[376]8.8037,[377]8.8148,[378]8.8284,[379]8.8390,[380]8.8577,[381]8.8731,[382]8.8873,[383]8.8997,[384]8.9121,[385]8.9398,[386]8.9572,[387]8.9592,[388]8.9606,[389]8.9685,[390]8.9916,[391]9.0118,[392]9.0092,[393]9.0068,[394]8.9998,[395]9.0002,[396]9.0093,[397]9.0136,[398]9.0151,[399]9.0200,[400]9.0358,[401]9.0392,[402]9.0389,[403]9.0288,[404]9.0197,[405]9.0086,[406]9.0037,[407]9.0078,[408]9.0125,[409]9.0091,[410]9.0078,[411]9.0194,[412]9.0216,[413]9.0181,[414]9.0109,[415]9.0000,[416]8.9862,[417]8.9881,[418]8.9888,[419]8.9870,[420]8.9827,[421]8.9848,[422]8.9703,[423]8.9702,[424]8.9664,[425]8.9640,[426]8.9660,[427]8.9734,[428]8.9854,[429]8.9910,[430]8.9852,[431]8.9782,[432]8.9832,[433]8.9821,[434]8.9803,[435]8.9902,[436]8.9772,[437]8.9787,[438]8.9782,[439]8.9701,[440]8.9795,[441]8.9758,[442]8.9677,[443]8.9611,[444]8.9618,[445]8.9508,[446]8.9544,[447]8.9521,[448]8.9426,[449]8.9333,[450]8.9348,[451]8.9310,[452]8.9178,[453]8.9090,[454]8.9044,[455]8.9046,[456]8.9018,[457]8.9067,[458]8.9228,[459]8.9191,[460]8.9195,[461]8.9148,[462]8.9139,[463]8.9248,[464]8.9240,[465]8.9264,[466]8.9284,[467]8.9341,[468]8.9408,[469]8.9442,[470]8.9504,[471]8.9408,[472]8.9479,[473]8.9368,[474]8.9379,[475]8.9455,[476]8.9491,[477]8.9436,[478]8.9316,[479]8.9338,[480]8.9431,[481]8.9505,[482]8.9400,[483]8.9486,[484]8.9555,[485]8.9582,[486]8.9558,[487]8.9607,[488]8.9523,[489]8.9391,[490]8.9381,[491]8.9294,[492]8.9257,[493]8.9140,[494]8.9104,[495]8.9025,[496]8.9010,[497]8.9099,[498]8.9157,[499]8.9101,[500]8.9104,[501]8.9127,[502]8.9097,[503]8.9239,[504]8.9318,[505]8.9347,[506]8.9338,[507]8.9276,[508]8.9323,[509]8.9259,[510]8.9275,[511]8.9330,[512]8.9285,[513]8.9306,[514]8.9337,[515]8.9353,[516]8.9378,[517]8.9438,[518]8.9422,[519]8.9414,[520]8.9405,[521]8.9424,[522]8.9330,[523]8.9372,[524]8.9372,[525]8.9398,[526]8.9454,[527]8.9458,[528]8.9462,[529]8.9425,[530]8.9378,[531]8.9411,[532]8.9373,[533]8.9376,[534]8.9375,[535]8.9406,[536]8.9342,[537]8.9426,[538]8.9535,[539]8.9526,[540]8.9681,[541]8.9682,[542]8.9586,[543]8.9606,[544]8.9674,[545]8.9644,[546]8.9625,[547]8.9558,[548]8.9420,[549]8.9398,[550]8.9259,[551]8.9152,[552]8.9047,[553]8.8751,[554]8.8725,[555]8.8753,[556]8.8766,[557]8.8767,[558]8.8752,[559]8.8801,[560]8.8846,[561]8.8912,[562]8.9031,[563]8.9110,[564]8.9083,[565]8.9173,[566]8.9215,[567]8.9116,[568]8.9038,[569]8.8974,[570]8.8955,[571]8.8936,[572]8.9040,[573]8.9069,[574]8.9099,[575]8.9105,[576]8.9181,[577]8.9149,[578]8.9194,[579]8.9272,[580]8.9403,[581]8.9417,[582]8.9526,[583]8.9375,[584]8.9344,
> Final estimate: PPL = 8.9344 +/- 0.06857
> ```
> 
> 👤 **ikawrakow** replied the **2025-05-02** at **06:28:45**:<br>
> So, this is a `0.0557` difference to the PPL I computed with Unsloth's imatrix, so about 0.6% higher. This is way too much to be explained by it being computed on different hardware (typically differences due to floating point operations non-associativity are in the 0.001 range for Wiki2 PPL). This would indicate
> * There is some level of numerical instability resulting in larger than usual differences between results computed on different hardware
> * And/Or `Q8_0` quantization of the KV cache is not accurate enough for this model (I used `fp16` KV cache).
> 
> If you have the ability and time to run with `fp16` KV cache, it would be interesting to have that result as well.
> 
> 👤 **saood06** replied the **2025-05-02** at **07:15:15**:<br>
> > If you have the ability and time to run with `fp16` KV cache, it would be interesting to have that result as well.
> 
> Here you go:
> 
> `.\llama-perplexity.exe -m "[...]" -ngl 99 -fa -fmoe  -f "[..]wiki.test.raw"`
> 
> ```
> [1]5.9736,[2]8.2473,[3]7.8248,[4]7.5090,[5]7.6181,[6]7.9293,[7]7.9364,[8]8.3848,[9]8.7403,[10]9.2418,[11]9.2909,[12]9.5013,[13]9.9446,[14]9.5539,[15]9.4583,[16]9.7112,[17]9.1785,[18]9.3274,[19]9.2756,[20]9.2376,[21]8.9346,[22]8.9130,[23]8.5527,[24]8.1115,[25]7.8918,[26]7.6819,[27]7.4769,[28]7.3594,[29]7.3887,[30]7.3210,[31]7.2760,[32]7.3085,[33]7.1979,[34]7.2693,[35]7.3798,[36]7.4989,[37]7.6636,[38]7.7104,[39]7.7082,[40]7.7709,[41]7.7718,[42]7.7430,[43]7.7996,[44]7.8064,[45]7.8028,[46]7.8195,[47]7.9951,[48]8.0858,[49]8.0708,[50]8.1428,[51]8.1885,[52]8.2210,[53]8.2908,[54]8.3622,[55]8.3631,[56]8.4006,[57]8.3778,[58]8.4120,[59]8.4784,[60]8.5297,[61]8.5523,[62]8.5996,[63]8.6570,[64]8.7165,[65]8.8000,[66]8.8702,[67]8.9450,[68]8.9290,[69]8.9443,[70]8.9367,[71]8.9676,[72]9.0349,[73]9.0725,[74]9.0898,[75]9.0438,[76]9.0420,[77]9.0893,[78]9.1437,[79]9.0755,[80]9.0532,[81]9.0161,[82]9.0547,[83]9.0174,[84]8.9989,[85]9.0142,[86]9.1062,[87]9.1386,[88]9.1295,[89]9.1316,[90]9.1027,[91]9.1535,[92]9.1270,[93]9.1707,[94]9.1783,[95]9.1520,[96]9.1371,[97]9.1107,[98]9.1225,[99]9.1011,[100]9.1586,[101]9.1854,[102]9.1710,[103]9.1861,[104]9.1507,[105]9.1506,[106]9.1385,[107]9.1680,[108]9.2054,[109]9.2309,[110]9.2824,[111]9.4013,[112]9.3940,[113]9.3502,[114]9.4151,[115]9.4236,[116]9.3861,[117]9.3689,[118]9.3468,[119]9.3020,[120]9.3153,[121]9.2962,[122]9.2866,[123]9.2463,[124]9.1913,[125]9.1611,[126]9.1364,[127]9.0790,[128]9.0479,[129]9.0107,[130]8.9785,[131]8.9274,[132]8.8878,[133]8.8715,[134]8.8671,[135]8.8644,[136]8.8624,[137]8.8311,[138]8.7971,[139]8.8032,[140]8.7863,[141]8.7775,[142]8.7831,[143]8.7891,[144]8.8182,[145]8.7880,[146]8.7468,[147]8.7090,[148]8.6709,[149]8.6509,[150]8.6152,[151]8.5858,[152]8.5716,[153]8.5746,[154]8.5300,[155]8.5322,[156]8.4904,[157]8.4639,[158]8.4276,[159]8.3930,[160]8.3574,[161]8.3378,[162]8.3230,[163]8.3033,[164]8.3020,[165]8.2830,[166]8.2750,[167]8.2675,[168]8.2916,[169]8.2929,[170]8.3244,[171]8.3492,[172]8.4016,[173]8.4460,[174]8.4650,[175]8.5220,[176]8.5503,[177]8.5972,[178]8.6381,[179]8.6490,[180]8.6490,[181]8.6707,[182]8.6941,[183]8.6781,[184]8.6918,[185]8.7008,[186]8.7041,[187]8.7146,[188]8.7149,[189]8.7290,[190]8.7586,[191]8.7662,[192]8.7813,[193]8.7756,[194]8.8000,[195]8.8202,[196]8.8308,[197]8.8377,[198]8.8159,[199]8.8094,[200]8.7984,[201]8.8067,[202]8.8194,[203]8.8394,[204]8.8558,[205]8.8719,[206]8.8608,[207]8.8861,[208]8.8674,[209]8.8694,[210]8.8649,[211]8.8664,[212]8.8664,[213]8.8654,[214]8.8467,[215]8.8299,[216]8.8243,[217]8.8339,[218]8.8314,[219]8.8026,[220]8.7723,[221]8.7606,[222]8.7501,[223]8.7457,[224]8.7614,[225]8.7408,[226]8.7417,[227]8.7322,[228]8.7067,[229]8.6769,[230]8.6544,[231]8.6385,[232]8.6238,[233]8.6229,[234]8.6317,[235]8.6266,[236]8.6119,[237]8.5987,[238]8.5786,[239]8.5695,[240]8.5717,[241]8.5735,[242]8.5840,[243]8.5833,[244]8.6012,[245]8.6028,[246]8.6218,[247]8.6262,[248]8.6298,[249]8.6374,[250]8.6450,[251]8.6627,[252]8.6781,[253]8.7056,[254]8.7244,[255]8.7280,[256]8.7426,[257]8.7539,[258]8.7389,[259]8.7176,[260]8.6965,[261]8.6721,[262]8.6554,[263]8.6484,[264]8.6496,[265]8.6603,[266]8.6635,[267]8.6630,[268]8.6536,[269]8.6558,[270]8.6532,[271]8.6436,[272]8.6434,[273]8.6404,[274]8.6366,[275]8.6370,[276]8.6256,[277]8.6240,[278]8.6297,[279]8.6269,[280]8.6228,[281]8.6157,[282]8.6225,[283]8.5953,[284]8.5628,[285]8.5682,[286]8.5529,[287]8.5352,[288]8.5319,[289]8.5352,[290]8.5574,[291]8.5611,[292]8.5600,[293]8.5597,[294]8.5771,[295]8.5966,[296]8.6104,[297]8.6343,[298]8.6301,[299]8.6159,[300]8.6174,[301]8.6142,[302]8.6097,[303]8.6022,[304]8.6197,[305]8.6192,[306]8.6158,[307]8.6179,[308]8.6149,[309]8.6137,[310]8.6182,[311]8.6222,[312]8.6118,[313]8.6043,[314]8.6079,[315]8.5949,[316]8.5993,[317]8.6204,[318]8.6258,[319]8.6203,[320]8.6228,[321]8.6086,[322]8.6199,[323]8.6346,[324]8.6507,[325]8.6710,[326]8.6732,[327]8.6655,[328]8.6653,[329]8.6499,[330]8.6404,[331]8.6312,[332]8.6335,[333]8.6336,[334]8.6258,[335]8.6146,[336]8.6087,[337]8.6148,[338]8.6240,[339]8.6169,[340]8.6086,[341]8.5971,[342]8.5949,[343]8.5896,[344]8.5983,[345]8.6018,[346]8.5975,[347]8.5856,[348]8.5863,[349]8.5795,[350]8.5734,[351]8.5733,[352]8.5784,[353]8.5782,[354]8.5653,[355]8.5821,[356]8.5903,[357]8.5962,[358]8.5844,[359]8.5856,[360]8.5831,[361]8.5881,[362]8.5832,[363]8.5793,[364]8.5895,[365]8.6065,[366]8.6323,[367]8.6493,[368]8.6782,[369]8.6979,[370]8.7129,[371]8.7341,[372]8.7573,[373]8.7651,[374]8.7751,[375]8.7954,[376]8.8094,[377]8.8205,[378]8.8340,[379]8.8444,[380]8.8630,[381]8.8783,[382]8.8923,[383]8.9046,[384]8.9177,[385]8.9451,[386]8.9627,[387]8.9649,[388]8.9664,[389]8.9747,[390]8.9977,[391]9.0179,[392]9.0151,[393]9.0123,[394]9.0053,[395]9.0057,[396]9.0149,[397]9.0193,[398]9.0209,[399]9.0254,[400]9.0412,[401]9.0448,[402]9.0445,[403]9.0342,[404]9.0250,[405]9.0143,[406]9.0092,[407]9.0131,[408]9.0179,[409]9.0147,[410]9.0133,[411]9.0250,[412]9.0270,[413]9.0237,[414]9.0164,[415]9.0056,[416]8.9918,[417]8.9939,[418]8.9943,[419]8.9925,[420]8.9881,[421]8.9901,[422]8.9757,[423]8.9752,[424]8.9716,[425]8.9691,[426]8.9713,[427]8.9781,[428]8.9900,[429]8.9958,[430]8.9898,[431]8.9829,[432]8.9878,[433]8.9866,[434]8.9847,[435]8.9947,[436]8.9817,[437]8.9833,[438]8.9826,[439]8.9745,[440]8.9837,[441]8.9798,[442]8.9722,[443]8.9657,[444]8.9664,[445]8.9557,[446]8.9592,[447]8.9568,[448]8.9472,[449]8.9380,[450]8.9395,[451]8.9356,[452]8.9220,[453]8.9135,[454]8.9089,[455]8.9092,[456]8.9065,[457]8.9113,[458]8.9274,[459]8.9238,[460]8.9241,[461]8.9196,[462]8.9185,[463]8.9295,[464]8.9291,[465]8.9318,[466]8.9338,[467]8.9392,[468]8.9456,[469]8.9488,[470]8.9550,[471]8.9455,[472]8.9530,[473]8.9420,[474]8.9434,[475]8.9509,[476]8.9546,[477]8.9489,[478]8.9368,[479]8.9392,[480]8.9484,[481]8.9561,[482]8.9454,[483]8.9540,[484]8.9609,[485]8.9638,[486]8.9614,[487]8.9661,[488]8.9577,[489]8.9444,[490]8.9436,[491]8.9348,[492]8.9310,[493]8.9193,[494]8.9158,[495]8.9076,[496]8.9063,[497]8.9151,[498]8.9211,[499]8.9155,[500]8.9159,[501]8.9183,[502]8.9154,[503]8.9297,[504]8.9373,[505]8.9398,[506]8.9389,[507]8.9328,[508]8.9376,[509]8.9313,[510]8.9331,[511]8.9384,[512]8.9338,[513]8.9362,[514]8.9392,[515]8.9409,[516]8.9433,[517]8.9492,[518]8.9474,[519]8.9465,[520]8.9458,[521]8.9477,[522]8.9383,[523]8.9423,[524]8.9424,[525]8.9450,[526]8.9508,[527]8.9511,[528]8.9515,[529]8.9478,[530]8.9430,[531]8.9463,[532]8.9421,[533]8.9426,[534]8.9426,[535]8.9459,[536]8.9394,[537]8.9478,[538]8.9587,[539]8.9576,[540]8.9731,[541]8.9730,[542]8.9633,[543]8.9653,[544]8.9722,[545]8.9691,[546]8.9674,[547]8.9609,[548]8.9473,[549]8.9452,[550]8.9316,[551]8.9211,[552]8.9108,[553]8.8812,[554]8.8786,[555]8.8814,[556]8.8827,[557]8.8827,[558]8.8813,[559]8.8863,[560]8.8909,[561]8.8975,[562]8.9095,[563]8.9175,[564]8.9143,[565]8.9233,[566]8.9277,[567]8.9180,[568]8.9102,[569]8.9038,[570]8.9022,[571]8.9006,[572]8.9107,[573]8.9135,[574]8.9165,[575]8.9171,[576]8.9246,[577]8.9213,[578]8.9259,[579]8.9338,[580]8.9469,[581]8.9482,[582]8.9594,[583]8.9442,[584]8.9408,
> Final estimate: PPL = 8.9408 +/- 0.06868
> ```
> 
> 👤 **ikawrakow** replied the **2025-05-02** at **08:08:40**:<br>
> Thanks.
> 
> This discards the second option and points more towards the first, given the `0.0065` difference between `Q8_0` and `fp16` KV cache on the *same hardware*. But there is also a 3rd option that I missed above:
> * There is (also) numerical instability is in the quantization process
> 
> I'm leaving for the airport shortly and will be traveling for the better part of the day. But tomorrow I'll post my `IQ4_KS` models quantized with the 3 different imatrix datasets on HF.
> 
> 👤 **danielhanchen** replied the **2025-05-02** at **08:12:08**:<br>
> @ikawrakow I think you're using the 128K imatrix which has YaRN enabled hence the discrepancy maybe. Also @ubergarm's results on Wiki show Q4_K_XL does pretty ok on Wiki.test.raw (Ubergarm's own quants look very impressive indeed), but higher on Ub's own calibration dataset. Notice I use Qwen's chat template directly, and add thinking traces so it might be worse on generic text data.
> 
> 👤 **saood06** replied the **2025-05-02** at **08:15:52**:<br>
> > This discards the second option and points more towards the first, given the `0.0065` difference between `Q8_0` and `fp16` KV cache on the _same hardware_. 
> 
> Do you want me to run PPL on that model on the CPU in my server, at FP16 and Q8_0? The model is fast enough for me to do that without it taking forever.
> 
> 👤 **ikawrakow** replied the **2025-05-02** at **08:19:31**:<br>
> > Do you want me to run PPL on that model on the CPU in my server, at FP16 and Q8_0?
> 
> That could be a useful datapoint. @ubergarm's `bf16` value differs from mine by more than I have historically found as a difference between different systems.
> 
> 👤 **ikawrakow** replied the **2025-05-02** at **08:37:24**:<br>
> @danielhanchen 
> 
> > I think you're using the 128K imatrix which has YaRN enabled hence...
> 
> So, what is the one I should use? I grabbed the first one I found (https://huggingface.co/unsloth/Qwen3-30B-A3B-128K-GGUF/blob/main/imatrix_unsloth.dat). But apart from this, why would YaRN enabled or not change anything when we are running a 512 tokens context, and the imatrix was not computed with a long context where YaRN would make a difference?
> 
> > Also @ubergarm's results on Wiki show Q4_K_XL does pretty ok on Wiki.test.raw 
> 
> This depends on the way you look at it. `PPL = 9.1688` is 1.1% higher than `bf16`, so pretty much a run-of-the-mill `Q4_K` quantization, especially for a MoE model (sometimes one needs `Q4_K_M` to get to the 1% range, but often `Q4_K_S` is enough). Your `IQ4_XS` quantization is actually better, arriving at essentially the same PPL (`9.1704`) with 1.3 GB smaller model size.
> 
> 👤 **danielhanchen** replied the **2025-05-02** at **08:48:36**:<br>
> @ikawrakow Oh my calibration dataset is like 12K or longer for thinking models, so there might be some discrepancies for 128K long context imatrices. https://blog.eleuther.ai/yarn/ for eg does show YaRN enabled does increase PPL for shorter context lengths.
> 
> Oh this one: https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/blob/main/imatrix_unsloth.dat (normal 40960 context length)
> 
> I used BF16 to get the imatrix. I actually tried Q8_0, and it actually failed to create some IQ1_S / IQ1_M quants, so I instead used BF16 - I think Qwen released FP8 versions, so I first thought using Q8_0 was fine for imatrix, since they might have trained with FP8, but I'm not sure anymore - the FP8 might just be post quantization.
> 
> I was actually thinking of adopting ik_llama.cpp @ikawrakow :) For the next release of models, I could provide quants also compatible with ik_llama.cpp if that's interesting, especially since @ubergarm's results always wow me (Deepseek, Scout, etc) :)
> 
> 👤 **saood06** replied the **2025-05-02** at **09:00:40**:<br>
> >But apart from this, why would YaRN enabled or not change anything when we are running a 512 tokens context
> 
> The official model card says:
> 
> "All the notable open-source frameworks implement static YaRN, which means the scaling factor remains constant regardless of input length, potentially impacting performance on shorter texts."
> 
> The whole reason I was doing PPL runs was to see if I could get dynamic YaRN working (see [this commit](https://github.com/ikawrakow/ik_llama.cpp/commit/a0d10704cd3982306da902dd460f9383ff919d1c)), and I thought testing PPL with high context values would be a good way to test it.
> 
> I had run this on my server:
> 
> `./bin/llama-perplexity -m /mnt/sda/Qwen3/30BA3B/BF16/ggml-model-IQ4_KS_R4.gguf -c 131072 -t 48 --numa distribute -fa -fmoe  -f /mnt/sda/wikitext-2-raw/wiki.test.raw -ctk q8_0 -ctv q8_0`
> 
> ```
> [1]6.7719,[2]7.1989,
> Final estimate: PPL = 7.1989 +/- 0.05142
> ```
> 
> (Note: `ggml-model-IQ4_KS_R4.gguf` is from repacking the mix I described above using `./bin/llama-quantize --repack /mnt/sda/Qwen3/30BA3B/BF16/ggml-model-IQ4_KS.gguf /mnt/sda/Qwen3/30BA3B/BF16/ggml-model-IQ4_KS_R4.gguf Q8_0`) 
> 
> But testing on the server took too long and I couldn't easily fit 128K on my GPU so I tested on the GPU using:
> 
>  `.\llama-perplexity.exe -m "[...]ggml-model-IQ4_KS.gguf" -ngl 99 -fa -fmoe  -f "[...]wiki.test.raw" -ctk iq4_nl -ctv iq4_nl -c 64000`
> 
> ```
> [1]8.8738,[2]8.0346,[3]8.2270,[4]8.1283,
> Final estimate: PPL = 8.1283 +/- 0.06022
> ```
> 
> It gave me the same result both times (my commit and main) so I'm not sure if my change did anything at all.
> 
> 👤 **ikawrakow** replied the **2025-05-03** at **04:41:26**:<br>
> > For the next release of models, I could provide quants also compatible with ik_llama.cpp if that's interesting,
> 
> This would be great!
> 
> 👤 **ikawrakow** replied the **2025-05-03** at **06:44:59**:<br>
> @saood06 Where did you get the RoPE factor change from?
> 
> 👤 **saood06** replied the **2025-05-03** at **06:49:45**:<br>
> Sorry for the delay but here is the same model (and it's repacked variant) running PPL on my CPU instead of my GPU with both F16 and Q8_0 cache. 
> 
> ` ./bin/llama-perplexity -m /mnt/sda/Qwen3/30BA3B/BF16/ggml-model-IQ4_KS.gguf  -t 48 --numa distribute -fa -fmoe  -f /mnt/sda/wikitext-2-raw/wiki.test.raw`
> 
> ```
> [1]5.9421,[2]8.1927,[3]7.7555,[4]7.4395,[5]7.5550,[6]7.8861,[7]7.8997,[8]8.3538,[9]8.6928,[10]9.2011,[11]9.2374,[12]9.4389,[13]9.8798,[14]9.4990,[15]9.4075,[16]9.6493,[17]9.1190,[18]9.2654,[19]9.2127,[20]9.1786,[21]8.8810,[22]8.8592,[23]8.4972,[24]8.0599,[25]7.8473,[26]7.6375,[27]7.4392,[28]7.3235,[29]7.3546,[30]7.2892,[31]7.2458,[32]7.2848,[33]7.1788,[34]7.2525,[35]7.3712,[36]7.4891,[37]7.6488,[38]7.6965,[39]7.6930,[40]7.7501,[41]7.7511,[42]7.7227,[43]7.7793,[44]7.7816,[45]7.7782,[46]7.7920,[47]7.9712,[48]8.0624,[49]8.0477,[50]8.1206,[51]8.1634,[52]8.1939,[53]8.2635,[54]8.3355,[55]8.3368,[56]8.3736,[57]8.3533,[58]8.3895,[59]8.4563,[60]8.5081,[61]8.5273,[62]8.5757,[63]8.6332,[64]8.6948,[65]8.7780,[66]8.8502,[67]8.9238,[68]8.9077,[69]8.9213,[70]8.9152,[71]8.9446,[72]9.0097,[73]9.0484,[74]9.0658,[75]9.0213,[76]9.0207,[77]9.0674,[78]9.1220,[79]9.0543,[80]9.0327,[81]8.9955,[82]9.0330,[83]8.9980,[84]8.9797,[85]8.9957,[86]9.0879,[87]9.1206,[88]9.1114,[89]9.1156,[90]9.0878,[91]9.1366,[92]9.1114,[93]9.1541,[94]9.1613,[95]9.1370,[96]9.1217,[97]9.0913,[98]9.1025,[99]9.0818,[100]9.1397,[101]9.1669,[102]9.1511,[103]9.1666,[104]9.1319,[105]9.1313,[106]9.1162,[107]9.1455,[108]9.1829,[109]9.2087,[110]9.2591,[111]9.3767,[112]9.3708,[113]9.3269,[114]9.3914,[115]9.4002,[116]9.3635,[117]9.3469,[118]9.3245,[119]9.2799,[120]9.2920,[121]9.2728,[122]9.2644,[123]9.2229,[124]9.1683,[125]9.1374,[126]9.1121,[127]9.0553,[128]9.0235,[129]8.9870,[130]8.9534,[131]8.9024,[132]8.8621,[133]8.8458,[134]8.8409,[135]8.8381,[136]8.8353,[137]8.8056,[138]8.7716,[139]8.7776,[140]8.7609,[141]8.7523,[142]8.7583,[143]8.7660,[144]8.7938,[145]8.7642,[146]8.7244,[147]8.6863,[148]8.6483,[149]8.6290,[150]8.5933,[151]8.5642,[152]8.5501,[153]8.5518,[154]8.5073,[155]8.5085,[156]8.4679,[157]8.4410,[158]8.4047,[159]8.3706,[160]8.3344,[161]8.3151,[162]8.2999,[163]8.2803,[164]8.2795,[165]8.2596,[166]8.2519,[167]8.2440,[168]8.2684,[169]8.2696,[170]8.3013,[171]8.3252,[172]8.3771,[173]8.4215,[174]8.4405,[175]8.4971,[176]8.5253,[177]8.5726,[178]8.6124,[179]8.6233,[180]8.6229,[181]8.6453,[182]8.6689,[183]8.6529,[184]8.6659,[185]8.6748,[186]8.6781,[187]8.6886,[188]8.6896,[189]8.7028,[190]8.7323,[191]8.7405,[192]8.7552,[193]8.7495,[194]8.7734,[195]8.7933,[196]8.8040,[197]8.8102,[198]8.7884,[199]8.7820,[200]8.7713,[201]8.7798,[202]8.7925,[203]8.8131,[204]8.8292,[205]8.8452,[206]8.8340,[207]8.8595,[208]8.8408,[209]8.8426,[210]8.8404,[211]8.8406,[212]8.8410,[213]8.8402,[214]8.8219,[215]8.8045,[216]8.7990,[217]8.8092,[218]8.8056,[219]8.7769,[220]8.7465,[221]8.7359,[222]8.7253,[223]8.7204,[224]8.7356,[225]8.7143,[226]8.7154,[227]8.7056,[228]8.6803,[229]8.6504,[230]8.6280,[231]8.6113,[232]8.5968,[233]8.5955,[234]8.6037,[235]8.5983,[236]8.5845,[237]8.5714,[238]8.5516,[239]8.5431,[240]8.5465,[241]8.5474,[242]8.5579,[243]8.5570,[244]8.5749,[245]8.5769,[246]8.5958,[247]8.6000,[248]8.6038,[249]8.6106,[250]8.6185,[251]8.6362,[252]8.6513,[253]8.6782,[254]8.6970,[255]8.7010,[256]8.7152,[257]8.7257,[258]8.7110,[259]8.6900,[260]8.6689,[261]8.6443,[262]8.6275,[263]8.6215,[264]8.6231,[265]8.6331,[266]8.6370,[267]8.6359,[268]8.6263,[269]8.6286,[270]8.6263,[271]8.6169,[272]8.6157,[273]8.6124,[274]8.6088,[275]8.6094,[276]8.5980,[277]8.5958,[278]8.6014,[279]8.5984,[280]8.5937,[281]8.5862,[282]8.5940,[283]8.5661,[284]8.5335,[285]8.5387,[286]8.5233,[287]8.5050,[288]8.5009,[289]8.5044,[290]8.5271,[291]8.5305,[292]8.5298,[293]8.5299,[294]8.5467,[295]8.5656,[296]8.5790,[297]8.6023,[298]8.5983,[299]8.5836,[300]8.5851,[301]8.5821,[302]8.5790,[303]8.5716,[304]8.5895,[305]8.5893,[306]8.5858,[307]8.5882,[308]8.5861,[309]8.5847,[310]8.5902,[311]8.5947,[312]8.5843,[313]8.5767,[314]8.5806,[315]8.5679,[316]8.5722,[317]8.5937,[318]8.5988,[319]8.5935,[320]8.5962,[321]8.5822,[322]8.5938,[323]8.6081,[324]8.6246,[325]8.6451,[326]8.6473,[327]8.6393,[328]8.6400,[329]8.6248,[330]8.6160,[331]8.6067,[332]8.6094,[333]8.6093,[334]8.6015,[335]8.5906,[336]8.5850,[337]8.5913,[338]8.6002,[339]8.5931,[340]8.5843,[341]8.5726,[342]8.5705,[343]8.5651,[344]8.5739,[345]8.5767,[346]8.5722,[347]8.5604,[348]8.5609,[349]8.5540,[350]8.5479,[351]8.5481,[352]8.5527,[353]8.5525,[354]8.5395,[355]8.5559,[356]8.5639,[357]8.5694,[358]8.5577,[359]8.5585,[360]8.5562,[361]8.5612,[362]8.5565,[363]8.5525,[364]8.5621,[365]8.5795,[366]8.6048,[367]8.6218,[368]8.6509,[369]8.6707,[370]8.6858,[371]8.7065,[372]8.7295,[373]8.7375,[374]8.7475,[375]8.7676,[376]8.7812,[377]8.7919,[378]8.8057,[379]8.8162,[380]8.8345,[381]8.8494,[382]8.8632,[383]8.8752,[384]8.8877,[385]8.9149,[386]8.9328,[387]8.9352,[388]8.9365,[389]8.9445,[390]8.9669,[391]8.9880,[392]8.9852,[393]8.9824,[394]8.9757,[395]8.9762,[396]8.9852,[397]8.9897,[398]8.9920,[399]8.9966,[400]9.0130,[401]9.0166,[402]9.0160,[403]9.0058,[404]8.9966,[405]8.9859,[406]8.9811,[407]8.9847,[408]8.9888,[409]8.9858,[410]8.9847,[411]8.9964,[412]8.9986,[413]8.9954,[414]8.9885,[415]8.9781,[416]8.9648,[417]8.9667,[418]8.9676,[419]8.9658,[420]8.9613,[421]8.9631,[422]8.9492,[423]8.9490,[424]8.9455,[425]8.9432,[426]8.9452,[427]8.9520,[428]8.9640,[429]8.9695,[430]8.9636,[431]8.9566,[432]8.9619,[433]8.9607,[434]8.9588,[435]8.9688,[436]8.9559,[437]8.9575,[438]8.9572,[439]8.9490,[440]8.9583,[441]8.9546,[442]8.9467,[443]8.9401,[444]8.9412,[445]8.9305,[446]8.9344,[447]8.9319,[448]8.9223,[449]8.9132,[450]8.9145,[451]8.9106,[452]8.8972,[453]8.8887,[454]8.8840,[455]8.8843,[456]8.8816,[457]8.8868,[458]8.9028,[459]8.8996,[460]8.8998,[461]8.8956,[462]8.8948,[463]8.9059,[464]8.9055,[465]8.9082,[466]8.9101,[467]8.9155,[468]8.9215,[469]8.9251,[470]8.9314,[471]8.9222,[472]8.9301,[473]8.9192,[474]8.9196,[475]8.9270,[476]8.9310,[477]8.9254,[478]8.9135,[479]8.9158,[480]8.9246,[481]8.9317,[482]8.9211,[483]8.9296,[484]8.9367,[485]8.9395,[486]8.9368,[487]8.9417,[488]8.9333,[489]8.9202,[490]8.9191,[491]8.9106,[492]8.9070,[493]8.8952,[494]8.8916,[495]8.8835,[496]8.8818,[497]8.8905,[498]8.8965,[499]8.8913,[500]8.8920,[501]8.8942,[502]8.8910,[503]8.9055,[504]8.9131,[505]8.9160,[506]8.9148,[507]8.9086,[508]8.9131,[509]8.9069,[510]8.9083,[511]8.9134,[512]8.9092,[513]8.9116,[514]8.9145,[515]8.9163,[516]8.9187,[517]8.9250,[518]8.9234,[519]8.9224,[520]8.9215,[521]8.9238,[522]8.9147,[523]8.9188,[524]8.9188,[525]8.9215,[526]8.9270,[527]8.9271,[528]8.9276,[529]8.9231,[530]8.9185,[531]8.9215,[532]8.9175,[533]8.9182,[534]8.9183,[535]8.9215,[536]8.9150,[537]8.9236,[538]8.9343,[539]8.9334,[540]8.9489,[541]8.9492,[542]8.9396,[543]8.9415,[544]8.9484,[545]8.9451,[546]8.9434,[547]8.9368,[548]8.9231,[549]8.9212,[550]8.9072,[551]8.8968,[552]8.8867,[553]8.8572,[554]8.8546,[555]8.8574,[556]8.8589,[557]8.8588,[558]8.8577,[559]8.8627,[560]8.8673,[561]8.8740,[562]8.8856,[563]8.8937,[564]8.8904,[565]8.8993,[566]8.9036,[567]8.8936,[568]8.8860,[569]8.8794,[570]8.8776,[571]8.8760,[572]8.8860,[573]8.8888,[574]8.8917,[575]8.8922,[576]8.9001,[577]8.8969,[578]8.9013,[579]8.9089,[580]8.9221,[581]8.9237,[582]8.9349,[583]8.9200,[584]8.9170,
> Final estimate: PPL = 8.9170 +/- 0.06824
> ```
> 
> 
> `./bin/llama-perplexity -m /mnt/sda/Qwen3/30BA3B/BF16/ggml-model-IQ4_KS.gguf  -t 48 --numa distribute -fa -fmoe  -f /mnt/sda/wikitext-2-raw/wiki.test.raw  -ctk q8_0 -ctv q8_0`
> 
> 
> ```
> [1]5.9698,[2]8.2139,[3]7.7738,[4]7.4546,[5]7.5752,[6]7.9056,[7]7.9415,[8]8.3840,[9]8.7487,[10]9.2367,[11]9.2841,[12]9.4800,[13]9.9219,[14]9.5403,[15]9.4546,[16]9.6851,[17]9.1534,[18]9.3022,[19]9.2445,[20]9.2133,[21]8.9134,[22]8.8940,[23]8.5302,[24]8.0827,[25]7.8647,[26]7.6599,[27]7.4641,[28]7.3532,[29]7.3885,[30]7.3247,[31]7.2747,[32]7.3120,[33]7.2107,[34]7.2830,[35]7.3981,[36]7.5158,[37]7.6756,[38]7.7215,[39]7.7115,[40]7.7688,[41]7.7703,[42]7.7393,[43]7.7965,[44]7.7986,[45]7.7936,[46]7.8104,[47]7.9858,[48]8.0737,[49]8.0606,[50]8.1330,[51]8.1770,[52]8.2086,[53]8.2796,[54]8.3494,[55]8.3499,[56]8.3851,[57]8.3638,[58]8.3999,[59]8.4671,[60]8.5179,[61]8.5405,[62]8.5877,[63]8.6435,[64]8.7023,[65]8.7852,[66]8.8551,[67]8.9286,[68]8.9145,[69]8.9277,[70]8.9194,[71]8.9510,[72]9.0171,[73]9.0557,[74]9.0724,[75]9.0272,[76]9.0225,[77]9.0650,[78]9.1172,[79]9.0464,[80]9.0257,[81]8.9900,[82]9.0252,[83]8.9890,[84]8.9700,[85]8.9875,[86]9.0793,[87]9.1106,[88]9.1015,[89]9.1045,[90]9.0749,[91]9.1245,[92]9.0968,[93]9.1382,[94]9.1442,[95]9.1199,[96]9.1039,[97]9.0760,[98]9.0876,[99]9.0659,[100]9.1233,[101]9.1498,[102]9.1372,[103]9.1525,[104]9.1179,[105]9.1163,[106]9.1045,[107]9.1354,[108]9.1735,[109]9.1986,[110]9.2487,[111]9.3663,[112]9.3599,[113]9.3166,[114]9.3820,[115]9.3925,[116]9.3553,[117]9.3387,[118]9.3151,[119]9.2699,[120]9.2806,[121]9.2612,[122]9.2534,[123]9.2131,[124]9.1587,[125]9.1269,[126]9.1023,[127]9.0467,[128]9.0160,[129]8.9811,[130]8.9475,[131]8.8969,[132]8.8577,[133]8.8414,[134]8.8370,[135]8.8345,[136]8.8324,[137]8.8022,[138]8.7676,[139]8.7736,[140]8.7571,[141]8.7484,[142]8.7534,[143]8.7607,[144]8.7886,[145]8.7587,[146]8.7188,[147]8.6810,[148]8.6444,[149]8.6255,[150]8.5908,[151]8.5611,[152]8.5463,[153]8.5492,[154]8.5054,[155]8.5070,[156]8.4666,[157]8.4402,[158]8.4033,[159]8.3682,[160]8.3315,[161]8.3114,[162]8.2962,[163]8.2771,[164]8.2760,[165]8.2558,[166]8.2478,[167]8.2405,[168]8.2627,[169]8.2631,[170]8.2947,[171]8.3201,[172]8.3730,[173]8.4162,[174]8.4354,[175]8.4923,[176]8.5212,[177]8.5677,[178]8.6082,[179]8.6206,[180]8.6200,[181]8.6437,[182]8.6669,[183]8.6507,[184]8.6640,[185]8.6732,[186]8.6761,[187]8.6859,[188]8.6859,[189]8.6987,[190]8.7283,[191]8.7363,[192]8.7514,[193]8.7456,[194]8.7695,[195]8.7891,[196]8.8005,[197]8.8075,[198]8.7860,[199]8.7794,[200]8.7682,[201]8.7765,[202]8.7886,[203]8.8081,[204]8.8242,[205]8.8400,[206]8.8288,[207]8.8542,[208]8.8358,[209]8.8391,[210]8.8362,[211]8.8367,[212]8.8362,[213]8.8354,[214]8.8169,[215]8.8006,[216]8.7962,[217]8.8056,[218]8.8027,[219]8.7743,[220]8.7441,[221]8.7326,[222]8.7217,[223]8.7167,[224]8.7314,[225]8.7104,[226]8.7108,[227]8.7013,[228]8.6756,[229]8.6456,[230]8.6240,[231]8.6082,[232]8.5933,[233]8.5914,[234]8.6001,[235]8.5946,[236]8.5809,[237]8.5680,[238]8.5473,[239]8.5396,[240]8.5420,[241]8.5438,[242]8.5542,[243]8.5537,[244]8.5714,[245]8.5738,[246]8.5927,[247]8.5967,[248]8.6000,[249]8.6075,[250]8.6154,[251]8.6335,[252]8.6486,[253]8.6756,[254]8.6940,[255]8.6984,[256]8.7128,[257]8.7238,[258]8.7091,[259]8.6880,[260]8.6664,[261]8.6413,[262]8.6248,[263]8.6184,[264]8.6197,[265]8.6298,[266]8.6328,[267]8.6319,[268]8.6224,[269]8.6243,[270]8.6226,[271]8.6127,[272]8.6116,[273]8.6088,[274]8.6052,[275]8.6053,[276]8.5936,[277]8.5915,[278]8.5968,[279]8.5941,[280]8.5890,[281]8.5821,[282]8.5899,[283]8.5621,[284]8.5293,[285]8.5347,[286]8.5203,[287]8.5017,[288]8.4975,[289]8.5006,[290]8.5231,[291]8.5268,[292]8.5259,[293]8.5258,[294]8.5428,[295]8.5615,[296]8.5757,[297]8.5991,[298]8.5948,[299]8.5804,[300]8.5820,[301]8.5789,[302]8.5754,[303]8.5677,[304]8.5851,[305]8.5850,[306]8.5815,[307]8.5840,[308]8.5813,[309]8.5798,[310]8.5848,[311]8.5891,[312]8.5793,[313]8.5720,[314]8.5759,[315]8.5632,[316]8.5675,[317]8.5888,[318]8.5941,[319]8.5888,[320]8.5913,[321]8.5775,[322]8.5893,[323]8.6036,[324]8.6202,[325]8.6407,[326]8.6434,[327]8.6356,[328]8.6357,[329]8.6207,[330]8.6117,[331]8.6021,[332]8.6055,[333]8.6056,[334]8.5974,[335]8.5865,[336]8.5808,[337]8.5873,[338]8.5965,[339]8.5896,[340]8.5814,[341]8.5697,[342]8.5676,[343]8.5621,[344]8.5707,[345]8.5741,[346]8.5693,[347]8.5572,[348]8.5582,[349]8.5512,[350]8.5455,[351]8.5457,[352]8.5501,[353]8.5494,[354]8.5369,[355]8.5534,[356]8.5616,[357]8.5670,[358]8.5552,[359]8.5561,[360]8.5539,[361]8.5589,[362]8.5541,[363]8.5512,[364]8.5610,[365]8.5775,[366]8.6031,[367]8.6201,[368]8.6491,[369]8.6687,[370]8.6833,[371]8.7041,[372]8.7271,[373]8.7348,[374]8.7444,[375]8.7646,[376]8.7779,[377]8.7884,[378]8.8020,[379]8.8125,[380]8.8307,[381]8.8463,[382]8.8603,[383]8.8728,[384]8.8855,[385]8.9130,[386]8.9304,[387]8.9322,[388]8.9337,[389]8.9417,[390]8.9643,[391]8.9847,[392]8.9817,[393]8.9783,[394]8.9711,[395]8.9715,[396]8.9809,[397]8.9852,[398]8.9871,[399]8.9918,[400]9.0075,[401]9.0107,[402]9.0104,[403]9.0007,[404]8.9916,[405]8.9805,[406]8.9757,[407]8.9795,[408]8.9844,[409]8.9810,[410]8.9800,[411]8.9920,[412]8.9944,[413]8.9914,[414]8.9845,[415]8.9740,[416]8.9605,[417]8.9625,[418]8.9636,[419]8.9620,[420]8.9577,[421]8.9596,[422]8.9456,[423]8.9455,[424]8.9415,[425]8.9393,[426]8.9407,[427]8.9476,[428]8.9594,[429]8.9649,[430]8.9589,[431]8.9517,[432]8.9571,[433]8.9560,[434]8.9537,[435]8.9636,[436]8.9511,[437]8.9527,[438]8.9522,[439]8.9444,[440]8.9540,[441]8.9500,[442]8.9419,[443]8.9348,[444]8.9359,[445]8.9252,[446]8.9292,[447]8.9264,[448]8.9164,[449]8.9077,[450]8.9092,[451]8.9051,[452]8.8917,[453]8.8833,[454]8.8785,[455]8.8788,[456]8.8766,[457]8.8814,[458]8.8975,[459]8.8939,[460]8.8943,[461]8.8896,[462]8.8884,[463]8.8989,[464]8.8984,[465]8.9008,[466]8.9031,[467]8.9086,[468]8.9151,[469]8.9190,[470]8.9254,[471]8.9158,[472]8.9231,[473]8.9122,[474]8.9134,[475]8.9210,[476]8.9246,[477]8.9192,[478]8.9072,[479]8.9092,[480]8.9184,[481]8.9254,[482]8.9152,[483]8.9239,[484]8.9308,[485]8.9337,[486]8.9312,[487]8.9362,[488]8.9279,[489]8.9150,[490]8.9147,[491]8.9061,[492]8.9022,[493]8.8903,[494]8.8867,[495]8.8788,[496]8.8775,[497]8.8858,[498]8.8921,[499]8.8864,[500]8.8867,[501]8.8891,[502]8.8860,[503]8.9003,[504]8.9081,[505]8.9108,[506]8.9098,[507]8.9038,[508]8.9080,[509]8.9018,[510]8.9035,[511]8.9086,[512]8.9043,[513]8.9066,[514]8.9095,[515]8.9112,[516]8.9138,[517]8.9199,[518]8.9183,[519]8.9174,[520]8.9168,[521]8.9190,[522]8.9097,[523]8.9138,[524]8.9138,[525]8.9163,[526]8.9218,[527]8.9217,[528]8.9220,[529]8.9179,[530]8.9131,[531]8.9162,[532]8.9123,[533]8.9130,[534]8.9128,[535]8.9159,[536]8.9096,[537]8.9177,[538]8.9283,[539]8.9273,[540]8.9430,[541]8.9428,[542]8.9333,[543]8.9355,[544]8.9427,[545]8.9394,[546]8.9374,[547]8.9307,[548]8.9169,[549]8.9150,[550]8.9012,[551]8.8905,[552]8.8801,[553]8.8505,[554]8.8478,[555]8.8509,[556]8.8522,[557]8.8522,[558]8.8509,[559]8.8561,[560]8.8605,[561]8.8672,[562]8.8788,[563]8.8865,[564]8.8834,[565]8.8924,[566]8.8963,[567]8.8861,[568]8.8786,[569]8.8717,[570]8.8696,[571]8.8680,[572]8.8781,[573]8.8812,[574]8.8843,[575]8.8849,[576]8.8931,[577]8.8898,[578]8.8943,[579]8.9022,[580]8.9153,[581]8.9168,[582]8.9279,[583]8.9129,[584]8.9098,
> Final estimate: PPL = 8.9098 +/- 0.06813
> ```
> 
> 
> `./bin/llama-perplexity -m /mnt/sda/Qwen3/30BA3B/BF16/ggml-model-IQ4_KS_R4.gguf  -t 48 --numa distribute -fa -fmoe  -f /mnt/sda/wikitext-2-raw/wiki.test.raw`
> 
> ```
> [1]5.9157,[2]8.2473,[3]7.8220,[4]7.4939,[5]7.6001,[6]7.9028,[7]7.9229,[8]8.3807,[9]8.7390,[10]9.2328,[11]9.2688,[12]9.4720,[13]9.9081,[14]9.5297,[15]9.4301,[16]9.6723,[17]9.1437,[18]9.2876,[19]9.2351,[20]9.1926,[21]8.8901,[22]8.8694,[23]8.5100,[24]8.0669,[25]7.8465,[26]7.6365,[27]7.4379,[28]7.3253,[29]7.3554,[30]7.2922,[31]7.2456,[32]7.2775,[33]7.1748,[34]7.2446,[35]7.3580,[36]7.4783,[37]7.6387,[38]7.6871,[39]7.6826,[40]7.7400,[41]7.7437,[42]7.7172,[43]7.7729,[44]7.7782,[45]7.7727,[46]7.7862,[47]7.9634,[48]8.0546,[49]8.0397,[50]8.1127,[51]8.1546,[52]8.1880,[53]8.2591,[54]8.3306,[55]8.3336,[56]8.3690,[57]8.3474,[58]8.3826,[59]8.4488,[60]8.5003,[61]8.5228,[62]8.5709,[63]8.6296,[64]8.6880,[65]8.7701,[66]8.8408,[67]8.9141,[68]8.8999,[69]8.9137,[70]8.9056,[71]8.9346,[72]9.0009,[73]9.0406,[74]9.0589,[75]9.0118,[76]9.0099,[77]9.0545,[78]9.1086,[79]9.0397,[80]9.0167,[81]8.9818,[82]9.0207,[83]8.9864,[84]8.9676,[85]8.9822,[86]9.0764,[87]9.1087,[88]9.1009,[89]9.1035,[90]9.0748,[91]9.1235,[92]9.0961,[93]9.1397,[94]9.1471,[95]9.1229,[96]9.1070,[97]9.0786,[98]9.0899,[99]9.0680,[100]9.1258,[101]9.1523,[102]9.1380,[103]9.1546,[104]9.1212,[105]9.1191,[106]9.1065,[107]9.1362,[108]9.1735,[109]9.1996,[110]9.2495,[111]9.3670,[112]9.3603,[113]9.3166,[114]9.3823,[115]9.3907,[116]9.3535,[117]9.3360,[118]9.3124,[119]9.2678,[120]9.2798,[121]9.2613,[122]9.2524,[123]9.2108,[124]9.1560,[125]9.1253,[126]9.1008,[127]9.0431,[128]9.0121,[129]8.9749,[130]8.9426,[131]8.8919,[132]8.8538,[133]8.8386,[134]8.8341,[135]8.8314,[136]8.8288,[137]8.7982,[138]8.7644,[139]8.7701,[140]8.7539,[141]8.7449,[142]8.7508,[143]8.7591,[144]8.7873,[145]8.7586,[146]8.7182,[147]8.6803,[148]8.6430,[149]8.6225,[150]8.5876,[151]8.5578,[152]8.5438,[153]8.5466,[154]8.5019,[155]8.5038,[156]8.4624,[157]8.4358,[158]8.3998,[159]8.3661,[160]8.3297,[161]8.3102,[162]8.2952,[163]8.2750,[164]8.2737,[165]8.2536,[166]8.2465,[167]8.2389,[168]8.2632,[169]8.2637,[170]8.2948,[171]8.3196,[172]8.3720,[173]8.4160,[174]8.4348,[175]8.4909,[176]8.5192,[177]8.5657,[178]8.6056,[179]8.6168,[180]8.6162,[181]8.6378,[182]8.6611,[183]8.6444,[184]8.6584,[185]8.6671,[186]8.6704,[187]8.6805,[188]8.6811,[189]8.6943,[190]8.7242,[191]8.7321,[192]8.7467,[193]8.7411,[194]8.7661,[195]8.7860,[196]8.7964,[197]8.8032,[198]8.7818,[199]8.7756,[200]8.7642,[201]8.7727,[202]8.7853,[203]8.8055,[204]8.8221,[205]8.8382,[206]8.8279,[207]8.8534,[208]8.8351,[209]8.8367,[210]8.8337,[211]8.8349,[212]8.8342,[213]8.8331,[214]8.8146,[215]8.7975,[216]8.7917,[217]8.8014,[218]8.7987,[219]8.7711,[220]8.7409,[221]8.7295,[222]8.7180,[223]8.7132,[224]8.7283,[225]8.7069,[226]8.7080,[227]8.6983,[228]8.6732,[229]8.6435,[230]8.6209,[231]8.6055,[232]8.5902,[233]8.5887,[234]8.5967,[235]8.5912,[236]8.5777,[237]8.5647,[238]8.5448,[239]8.5360,[240]8.5391,[241]8.5406,[242]8.5511,[243]8.5503,[244]8.5682,[245]8.5703,[246]8.5890,[247]8.5938,[248]8.5976,[249]8.6049,[250]8.6124,[251]8.6300,[252]8.6456,[253]8.6730,[254]8.6916,[255]8.6963,[256]8.7104,[257]8.7213,[258]8.7070,[259]8.6856,[260]8.6644,[261]8.6397,[262]8.6227,[263]8.6159,[264]8.6173,[265]8.6274,[266]8.6306,[267]8.6298,[268]8.6206,[269]8.6228,[270]8.6200,[271]8.6101,[272]8.6093,[273]8.6062,[274]8.6018,[275]8.6015,[276]8.5897,[277]8.5881,[278]8.5929,[279]8.5900,[280]8.5854,[281]8.5786,[282]8.5859,[283]8.5588,[284]8.5263,[285]8.5317,[286]8.5161,[287]8.4981,[288]8.4946,[289]8.4978,[290]8.5203,[291]8.5238,[292]8.5232,[293]8.5233,[294]8.5408,[295]8.5598,[296]8.5742,[297]8.5971,[298]8.5933,[299]8.5790,[300]8.5806,[301]8.5773,[302]8.5733,[303]8.5657,[304]8.5827,[305]8.5823,[306]8.5789,[307]8.5809,[308]8.5787,[309]8.5770,[310]8.5815,[311]8.5862,[312]8.5759,[313]8.5682,[314]8.5717,[315]8.5585,[316]8.5634,[317]8.5847,[318]8.5897,[319]8.5844,[320]8.5872,[321]8.5734,[322]8.5844,[323]8.5991,[324]8.6155,[325]8.6356,[326]8.6381,[327]8.6301,[328]8.6303,[329]8.6153,[330]8.6061,[331]8.5973,[332]8.5999,[333]8.5999,[334]8.5921,[335]8.5812,[336]8.5756,[337]8.5821,[338]8.5913,[339]8.5839,[340]8.5758,[341]8.5642,[342]8.5618,[343]8.5566,[344]8.5652,[345]8.5691,[346]8.5649,[347]8.5530,[348]8.5540,[349]8.5472,[350]8.5414,[351]8.5411,[352]8.5458,[353]8.5457,[354]8.5328,[355]8.5492,[356]8.5573,[357]8.5633,[358]8.5514,[359]8.5523,[360]8.5497,[361]8.5543,[362]8.5495,[363]8.5458,[364]8.5554,[365]8.5725,[366]8.5979,[367]8.6146,[368]8.6439,[369]8.6631,[370]8.6778,[371]8.6986,[372]8.7211,[373]8.7290,[374]8.7387,[375]8.7588,[376]8.7726,[377]8.7831,[378]8.7964,[379]8.8068,[380]8.8250,[381]8.8401,[382]8.8543,[383]8.8666,[384]8.8798,[385]8.9075,[386]8.9248,[387]8.9267,[388]8.9279,[389]8.9358,[390]8.9591,[391]8.9796,[392]8.9769,[393]8.9740,[394]8.9672,[395]8.9675,[396]8.9766,[397]8.9813,[398]8.9831,[399]8.9883,[400]9.0036,[401]9.0072,[402]9.0072,[403]8.9972,[404]8.9879,[405]8.9777,[406]8.9726,[407]8.9762,[408]8.9813,[409]8.9775,[410]8.9758,[411]8.9877,[412]8.9897,[413]8.9867,[414]8.9799,[415]8.9693,[416]8.9558,[417]8.9578,[418]8.9586,[419]8.9569,[420]8.9527,[421]8.9549,[422]8.9407,[423]8.9403,[424]8.9365,[425]8.9341,[426]8.9360,[427]8.9429,[428]8.9544,[429]8.9601,[430]8.9540,[431]8.9469,[432]8.9523,[433]8.9512,[434]8.9489,[435]8.9594,[436]8.9466,[437]8.9483,[438]8.9478,[439]8.9395,[440]8.9488,[441]8.9452,[442]8.9370,[443]8.9305,[444]8.9315,[445]8.9208,[446]8.9245,[447]8.9220,[448]8.9125,[449]8.9034,[450]8.9049,[451]8.9017,[452]8.8887,[453]8.8803,[454]8.8753,[455]8.8755,[456]8.8730,[457]8.8779,[458]8.8941,[459]8.8906,[460]8.8910,[461]8.8865,[462]8.8853,[463]8.8961,[464]8.8953,[465]8.8979,[466]8.8999,[467]8.9054,[468]8.9119,[469]8.9154,[470]8.9217,[471]8.9120,[472]8.9195,[473]8.9084,[474]8.9099,[475]8.9171,[476]8.9205,[477]8.9152,[478]8.9034,[479]8.9054,[480]8.9142,[481]8.9213,[482]8.9111,[483]8.9197,[484]8.9266,[485]8.9294,[486]8.9270,[487]8.9319,[488]8.9236,[489]8.9105,[490]8.9095,[491]8.9012,[492]8.8975,[493]8.8860,[494]8.8824,[495]8.8746,[496]8.8731,[497]8.8814,[498]8.8876,[499]8.8821,[500]8.8825,[501]8.8850,[502]8.8818,[503]8.8957,[504]8.9033,[505]8.9059,[506]8.9049,[507]8.8988,[508]8.9034,[509]8.8972,[510]8.8991,[511]8.9044,[512]8.9000,[513]8.9028,[514]8.9058,[515]8.9072,[516]8.9096,[517]8.9156,[518]8.9138,[519]8.9129,[520]8.9122,[521]8.9142,[522]8.9048,[523]8.9088,[524]8.9086,[525]8.9113,[526]8.9169,[527]8.9171,[528]8.9176,[529]8.9134,[530]8.9088,[531]8.9120,[532]8.9079,[533]8.9081,[534]8.9082,[535]8.9116,[536]8.9054,[537]8.9138,[538]8.9247,[539]8.9238,[540]8.9393,[541]8.9391,[542]8.9295,[543]8.9317,[544]8.9385,[545]8.9352,[546]8.9336,[547]8.9269,[548]8.9135,[549]8.9116,[550]8.8978,[551]8.8873,[552]8.8769,[553]8.8476,[554]8.8448,[555]8.8473,[556]8.8489,[557]8.8486,[558]8.8473,[559]8.8520,[560]8.8566,[561]8.8632,[562]8.8751,[563]8.8830,[564]8.8798,[565]8.8891,[566]8.8933,[567]8.8835,[568]8.8757,[569]8.8693,[570]8.8674,[571]8.8657,[572]8.8757,[573]8.8785,[574]8.8814,[575]8.8817,[576]8.8893,[577]8.8860,[578]8.8904,[579]8.8981,[580]8.9112,[581]8.9128,[582]8.9239,[583]8.9090,[584]8.9062,
> Final estimate: PPL = 8.9062 +/- 0.06811
> ```
> 
> `./bin/llama-perplexity -m /mnt/sda/Qwen3/30BA3B/BF16/ggml-model-IQ4_KS_R4.gguf  -t 48 --numa distribute -fa -fmoe  -f /mnt/sda/wikitext-2-raw/wiki.test.raw  -ctk q8_0 -ctv q8_0`
> 
> 
> ```
> [1]5.9016,[2]8.2000,[3]7.8140,[4]7.4725,[5]7.5629,[6]7.8727,[7]7.9035,[8]8.3263,[9]8.6767,[10]9.1767,[11]9.2245,[12]9.4083,[13]9.8413,[14]9.4664,[15]9.3727,[16]9.6102,[17]9.0849,[18]9.2388,[19]9.1735,[20]9.1432,[21]8.8470,[22]8.8168,[23]8.4654,[24]8.0293,[25]7.8182,[26]7.6125,[27]7.4136,[28]7.2990,[29]7.3292,[30]7.2632,[31]7.2147,[32]7.2425,[33]7.1391,[34]7.2118,[35]7.3291,[36]7.4505,[37]7.6066,[38]7.6595,[39]7.6571,[40]7.7174,[41]7.7206,[42]7.6952,[43]7.7506,[44]7.7565,[45]7.7513,[46]7.7658,[47]7.9396,[48]8.0275,[49]8.0138,[50]8.0853,[51]8.1307,[52]8.1643,[53]8.2358,[54]8.3060,[55]8.3053,[56]8.3430,[57]8.3225,[58]8.3581,[59]8.4309,[60]8.4806,[61]8.5032,[62]8.5485,[63]8.6048,[64]8.6630,[65]8.7482,[66]8.8175,[67]8.8924,[68]8.8787,[69]8.8915,[70]8.8836,[71]8.9148,[72]8.9826,[73]9.0235,[74]9.0416,[75]8.9940,[76]8.9918,[77]9.0371,[78]9.0880,[79]9.0183,[80]8.9964,[81]8.9587,[82]8.9944,[83]8.9610,[84]8.9402,[85]8.9564,[86]9.0486,[87]9.0821,[88]9.0735,[89]9.0747,[90]9.0454,[91]9.0955,[92]9.0694,[93]9.1150,[94]9.1229,[95]9.0985,[96]9.0840,[97]9.0566,[98]9.0688,[99]9.0485,[100]9.1061,[101]9.1346,[102]9.1200,[103]9.1360,[104]9.1011,[105]9.1018,[106]9.0895,[107]9.1188,[108]9.1568,[109]9.1825,[110]9.2341,[111]9.3521,[112]9.3452,[113]9.3028,[114]9.3675,[115]9.3770,[116]9.3409,[117]9.3241,[118]9.3013,[119]9.2564,[120]9.2678,[121]9.2488,[122]9.2393,[123]9.2002,[124]9.1469,[125]9.1159,[126]9.0924,[127]9.0347,[128]9.0034,[129]8.9666,[130]8.9353,[131]8.8850,[132]8.8451,[133]8.8305,[134]8.8248,[135]8.8234,[136]8.8209,[137]8.7910,[138]8.7567,[139]8.7631,[140]8.7471,[141]8.7386,[142]8.7445,[143]8.7522,[144]8.7807,[145]8.7505,[146]8.7112,[147]8.6732,[148]8.6360,[149]8.6164,[150]8.5807,[151]8.5516,[152]8.5375,[153]8.5405,[154]8.4962,[155]8.4992,[156]8.4582,[157]8.4322,[158]8.3958,[159]8.3613,[160]8.3245,[161]8.3042,[162]8.2882,[163]8.2678,[164]8.2657,[165]8.2456,[166]8.2368,[167]8.2301,[168]8.2538,[169]8.2544,[170]8.2859,[171]8.3102,[172]8.3630,[173]8.4064,[174]8.4245,[175]8.4801,[176]8.5084,[177]8.5554,[178]8.5952,[179]8.6057,[180]8.6065,[181]8.6286,[182]8.6513,[183]8.6357,[184]8.6486,[185]8.6579,[186]8.6620,[187]8.6730,[188]8.6738,[189]8.6871,[190]8.7167,[191]8.7243,[192]8.7397,[193]8.7343,[194]8.7577,[195]8.7769,[196]8.7888,[197]8.7953,[198]8.7731,[199]8.7664,[200]8.7559,[201]8.7647,[202]8.7775,[203]8.7980,[204]8.8137,[205]8.8302,[206]8.8196,[207]8.8455,[208]8.8264,[209]8.8284,[210]8.8242,[211]8.8247,[212]8.8245,[213]8.8242,[214]8.8063,[215]8.7894,[216]8.7850,[217]8.7952,[218]8.7927,[219]8.7639,[220]8.7339,[221]8.7223,[222]8.7116,[223]8.7064,[224]8.7211,[225]8.7003,[226]8.7008,[227]8.6912,[228]8.6655,[229]8.6352,[230]8.6134,[231]8.5975,[232]8.5829,[233]8.5819,[234]8.5907,[235]8.5855,[236]8.5715,[237]8.5586,[238]8.5385,[239]8.5304,[240]8.5332,[241]8.5353,[242]8.5460,[243]8.5450,[244]8.5628,[245]8.5646,[246]8.5836,[247]8.5875,[248]8.5915,[249]8.5992,[250]8.6072,[251]8.6246,[252]8.6396,[253]8.6664,[254]8.6849,[255]8.6886,[256]8.7028,[257]8.7138,[258]8.6980,[259]8.6768,[260]8.6553,[261]8.6303,[262]8.6140,[263]8.6072,[264]8.6084,[265]8.6184,[266]8.6213,[267]8.6203,[268]8.6109,[269]8.6128,[270]8.6113,[271]8.6013,[272]8.6002,[273]8.5974,[274]8.5942,[275]8.5939,[276]8.5820,[277]8.5805,[278]8.5852,[279]8.5820,[280]8.5774,[281]8.5707,[282]8.5787,[283]8.5512,[284]8.5188,[285]8.5241,[286]8.5089,[287]8.4909,[288]8.4869,[289]8.4895,[290]8.5114,[291]8.5149,[292]8.5140,[293]8.5142,[294]8.5316,[295]8.5506,[296]8.5642,[297]8.5877,[298]8.5841,[299]8.5692,[300]8.5706,[301]8.5680,[302]8.5639,[303]8.5569,[304]8.5745,[305]8.5746,[306]8.5707,[307]8.5734,[308]8.5716,[309]8.5707,[310]8.5753,[311]8.5795,[312]8.5694,[313]8.5621,[314]8.5663,[315]8.5533,[316]8.5577,[317]8.5790,[318]8.5843,[319]8.5790,[320]8.5820,[321]8.5679,[322]8.5800,[323]8.5938,[324]8.6105,[325]8.6308,[326]8.6332,[327]8.6251,[328]8.6260,[329]8.6108,[330]8.6016,[331]8.5926,[332]8.5956,[333]8.5957,[334]8.5876,[335]8.5771,[336]8.5711,[337]8.5770,[338]8.5861,[339]8.5789,[340]8.5703,[341]8.5586,[342]8.5562,[343]8.5509,[344]8.5587,[345]8.5623,[346]8.5576,[347]8.5456,[348]8.5460,[349]8.5389,[350]8.5331,[351]8.5330,[352]8.5379,[353]8.5370,[354]8.5242,[355]8.5401,[356]8.5479,[357]8.5531,[358]8.5408,[359]8.5419,[360]8.5397,[361]8.5451,[362]8.5403,[363]8.5366,[364]8.5463,[365]8.5628,[366]8.5886,[367]8.6058,[368]8.6349,[369]8.6541,[370]8.6687,[371]8.6894,[372]8.7124,[373]8.7204,[374]8.7299,[375]8.7500,[376]8.7637,[377]8.7737,[378]8.7876,[379]8.7978,[380]8.8164,[381]8.8318,[382]8.8456,[383]8.8580,[384]8.8710,[385]8.8986,[386]8.9157,[387]8.9177,[388]8.9189,[389]8.9269,[390]8.9499,[391]8.9701,[392]8.9670,[393]8.9642,[394]8.9573,[395]8.9575,[396]8.9663,[397]8.9706,[398]8.9717,[399]8.9767,[400]8.9924,[401]8.9957,[402]8.9955,[403]8.9853,[404]8.9763,[405]8.9653,[406]8.9603,[407]8.9638,[408]8.9684,[409]8.9650,[410]8.9634,[411]8.9757,[412]8.9778,[413]8.9744,[414]8.9673,[415]8.9566,[416]8.9433,[417]8.9448,[418]8.9459,[419]8.9444,[420]8.9400,[421]8.9419,[422]8.9279,[423]8.9277,[424]8.9237,[425]8.9214,[426]8.9228,[427]8.9294,[428]8.9414,[429]8.9471,[430]8.9413,[431]8.9345,[432]8.9398,[433]8.9388,[434]8.9367,[435]8.9467,[436]8.9339,[437]8.9353,[438]8.9351,[439]8.9267,[440]8.9365,[441]8.9333,[442]8.9257,[443]8.9191,[444]8.9199,[445]8.9096,[446]8.9130,[447]8.9104,[448]8.9008,[449]8.8919,[450]8.8932,[451]8.8895,[452]8.8761,[453]8.8675,[454]8.8629,[455]8.8631,[456]8.8609,[457]8.8660,[458]8.8819,[459]8.8785,[460]8.8785,[461]8.8742,[462]8.8731,[463]8.8837,[464]8.8831,[465]8.8852,[466]8.8872,[467]8.8925,[468]8.8989,[469]8.9025,[470]8.9087,[471]8.8997,[472]8.9071,[473]8.8964,[474]8.8979,[475]8.9053,[476]8.9089,[477]8.9035,[478]8.8919,[479]8.8941,[480]8.9030,[481]8.9105,[482]8.9003,[483]8.9091,[484]8.9163,[485]8.9196,[486]8.9168,[487]8.9217,[488]8.9133,[489]8.9002,[490]8.8997,[491]8.8910,[492]8.8872,[493]8.8755,[494]8.8719,[495]8.8641,[496]8.8628,[497]8.8714,[498]8.8780,[499]8.8725,[500]8.8728,[501]8.8752,[502]8.8721,[503]8.8863,[504]8.8940,[505]8.8969,[506]8.8959,[507]8.8901,[508]8.8949,[509]8.8884,[510]8.8903,[511]8.8956,[512]8.8914,[513]8.8939,[514]8.8972,[515]8.8990,[516]8.9014,[517]8.9072,[518]8.9054,[519]8.9049,[520]8.9039,[521]8.9059,[522]8.8965,[523]8.9003,[524]8.9001,[525]8.9028,[526]8.9086,[527]8.9088,[528]8.9095,[529]8.9053,[530]8.9005,[531]8.9036,[532]8.8999,[533]8.9003,[534]8.9000,[535]8.9031,[536]8.8967,[537]8.9052,[538]8.9158,[539]8.9149,[540]8.9305,[541]8.9301,[542]8.9205,[543]8.9225,[544]8.9292,[545]8.9262,[546]8.9243,[547]8.9178,[548]8.9041,[549]8.9023,[550]8.8888,[551]8.8781,[552]8.8676,[553]8.8381,[554]8.8353,[555]8.8383,[556]8.8396,[557]8.8397,[558]8.8384,[559]8.8437,[560]8.8485,[561]8.8554,[562]8.8673,[563]8.8752,[564]8.8724,[565]8.8813,[566]8.8853,[567]8.8753,[568]8.8676,[569]8.8609,[570]8.8589,[571]8.8573,[572]8.8675,[573]8.8705,[574]8.8733,[575]8.8736,[576]8.8814,[577]8.8783,[578]8.8829,[579]8.8908,[580]8.9042,[581]8.9057,[582]8.9168,[583]8.9019,[584]8.8990,
> Final estimate: PPL = 8.8990 +/- 0.06799
> ```
> 
> 👤 **ubergarm** replied the **2025-05-03** at **16:38:27**:<br>
> > > For the next release of models, I could provide quants also compatible with ik_llama.cpp if that's interesting,
> > 
> > This would be great!
> 
> Exciting times!
> 
> @danielhanchen fwiw myself and a few others have started adding the tag `ik_llama.cpp` to the `iqN_k` quants uploaded on the huggingface README.md model cards which makes it easier to find e.g. https://huggingface.co/models?other=ik_llama.cpp
> 
> Appreciate all your time and thoughtfulness lately with all the excitement haha... Cheers!
> 
> 👤 **saood06** replied the **2025-05-04** at **05:02:36**:<br>
> > @saood06 Where did you get the RoPE factor change from?
> 
> Nowhere, I was just experimenting after I saw that statement in the model card, but I didn't get very far.

---

👤 **ikawrakow** replied the **2025-05-03** at **06:12:51**:<br>

I have posted 3 `IQ4_KS` models quantized with the 3 different imatrix datasets discussed above [here](https://huggingface.co/ikawrakow/Qwen3-30B-A3B)

> 👤 **ubergarm** replied the **2025-05-04** at **04:21:54**:<br>
> I attempted to make a "visual diff" of three imatrix files. I didn't find yours @ikawrakow on the hf repo, so used mine, unsloths non-128k version, and bartowski's.
> 
> https://gist.github.com/ubergarm/2aa9327f7b98a9b16fef62b4941c7e76
> 
> I use @EAddario's mainline PR `--show-statistics` to print out a bunch of numbers and graph them as described briefly in the gist.
> 
> I'm not sure how to read the tea leaves or if this is just an amusing distraction and excuse to test vibe coding with `Qwen3-30B-A3B`. To make matters a bit more confusing, unsloth gave some details of their methodology which seems to include generating imatrix with larger context than the `-c 512` I use (and I assume is typical and default?). Its a useful comment in an otherwise odd discussion: https://huggingface.co/unsloth/Phi-4-reasoning-plus-GGUF/discussions/1#68152ae82c118dc537ae3667
> 
> Haven't had a chance to grab PPL and KLD stats on your 3 quants yet, but might be able to get that on Sunday and update my table above.
> 
> 👤 **ubergarm** replied the **2025-05-08** at **18:43:48**:<br>
> Just posted a quant roundup and trying to run some benchmarks against your Qwen3 `IQ4_KS` on hf. Already posted some PPL and KLD stats here: https://www.reddit.com/r/LocalLLaMA/comments/1khwxal/the_great_quant_wars_of_2025/
> 
> 👤 **l15y** replied the **2025-05-17** at **08:47:42**:<br>
> Please upload the IQK-3 version, which is very useful for users with 16G VRAM.

---

👤 **ikawrakow** replied the **2025-05-08** at **19:30:19**:<br>

@ubergarm Great write up!

The fact that the ikawrakow/IQ4_KS_Unsloth model gets a lower PPL than `bf16` on your private evaluation dataset is another indication that something is not quite right.

My only comment: when there is no doubt that the `bf16` model is best, then KLD and other token probability statistics against the predictions of the `bf16` model are great. But when one is not sure (see above), then KLD, etc., can be misleading. Suppose the following is true:
* They trained in some `fp4` variant
* Towards the end of the training, they decided to not show their cards just yet, and trained some more epochs in `bf16`
* Something didn't quite work out in these last iterations
* The released `bf16` model ended up being crippled
* By quantizing to something similar to `fp4`, one recovers a better quality model

In that scenario, the larger the difference between the `bf16` model and the (better) quantized model, the higher the values of KLD, etc. So that, if we went by these metrics, we would be thinking that the quantized model is not good, while in reality it is better.

> 👤 **saood06** replied the **2025-05-08** at **22:56:01**:<br>
> > @ubergarm Great write up!
> > 
> > The fact that the ikawrakow/IQ4_KS_Unsloth model gets a lower PPL than `bf16` on your private evaluation dataset is another indication that something is not quite right.
> 
> Something that I'm not sure that has been mentioned in this discussion is Qwen 3 states that only 2 of the base models went through the full post-training process the rest of the models in the family are distillations. Could it be that the odd results we are seeing might only impact the distilled models (as I can't find details on how they did the distillation)?
> 
> An interesting experiment would be to see if the odd results seen with Qwen3-30B-A3B can be reproduced with [Qwen3-30B-A3B-Base](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base).
> 
> See this graphic from their blog:
> 
> ![post-training-1](https://github.com/user-attachments/assets/39c3ef6d-3a9b-41a2-9bca-24649a0a9243)

---

👤 **afsara-ben** replied the **2025-06-10** at **22:24:55**:<br>

@ikawrakow i am having a hard time understanding how the iqx_k quants came from? is there an explanation somewhere other than the code

> 👤 **saood06** replied the **2025-06-11** at **02:58:40**:<br>
> #8 has the info you are looking for.