### 🗣️ [#288](https://github.com/ikawrakow/ik_llama.cpp/discussions/288) - On @compilade's PR 12557 and @jukofyork's quantization ideas

| **Author** | `ikawrakow` |
| :--- | :--- |
| **Created** | 2025-03-25 |
| **Updated** | 2025-04-11 |

---

#### Description

@compilade has submitted an [interesting PR](https://github.com/ggml-org/llama.cpp/pull/12557) in the mainline `llama.cpp` repository. As it is often the case, @jukofyork has improvement ideas. As both pinged me, and as I no longer hang around in the `llama.cpp` project, I'll address the pings here.

### @compilade's PR

First of all, this is a nice piece of work, so congratulations!

I did try the PR on a few models. I focused on `Q3_K` and `IQ4_NL` as I don't see the utility of using quantization types meant for ternary models (`TQ1_0`, `TQ2_0`) also for non-ternary models, and am also not particularly interested in the legacy quantization types (`Q4_0`, `Q5_0`, too low quality relative to the bits spent). I could have also looked at `IQ4_XS`, but it is very similar to `IQ4_NL`, so here we go with my observations:
* Without imatrix, the existing quantization methods are strictly better than your PR as measured by perplexity<sup>1</sup>
* With imatrix and pure quantization, your `Q3_K` is significantly better than the existing quantization method (but see below). `IQ4_NL` is hit-or-miss - sometimes slightly better, sometimes slightly worse, but overall not much of a difference apart from the 5X increase in quantization time.
* When I added the imatrix to `llama.cpp` it wasn't clear that it will take off the way it did. Hence, the quantization methods I contributed are the way they are. Perhaps they are suboptimal when there is a (meaningful) imatrix, but a major driving force was to make them as robust as possible for quantization without imatrix.
* I have run into this on a number of occasions when I was still actively working on quantization: in many models some tensors have a disproportionally high impact on the observed quantization quality. So, when using `--pure`, it may appear that one gets an improvement because the new method being tested happens to do better on exactly these tensors, but worse on many others. One gets excited about having improved things, but then in practice, with the high-impact tensors quantized with more bits in the quantization mix, suddenly the observed quality is lower than what one had before. Case in point, `Q3_K_M` with your PR often has a higher PPL than the existing quantization, despite being clearly better with `--pure`
* More on `--pure`: in some models token embedding quantization has a disproportional impact on observed quality, and some quantization types do not quantize `token_embd.weight` very well.  You do use `Q8_0` for the output tensor, I think it would be better to also use `Q8_0` for token embeddings when using `--pure`. 
* It is not that I didn't know how to implement exact minimization of RMSE (or maximization of cosine similarity, if that's what you prefer). The existing methods are the way they are because of the observation that the exact solution of the optimization problem often leads to disastrous results for observed quantization quality. RMSE (or cosine similarity) are just surrogates, so finding a better solution does not automatically lead to better quantization quality. I have seen people describe some of the k- and i-quant quantization methods as "brute force". They are not (brute force will look completely different and would take much longer. Also, the moment we decided to use brute force, that would be the moment where we would plug in an exact solution method that runs many times faster than brute force). They use carefully tuned heuristics to avoid the quants getting lost in the fields. When the iamtrix came along I was exited to use exact solution methods instead of heuristics. Unfortunately, even with an imatrix, one can (and often does) end up with a worse outcome with quantized weights that are more similar to the original model weights (as measured by the surrogate). 
* `IQ4_K` and `IQ5_K` here are miles ahead of any 4- or 5-bpw quantization type in mainline `llama.cpp`. Hence, I'm skeptical that they can be improved with your PR (but you are more than welcome to submit a PR here if you are able to demonstrate improvement). `IQ2_K` and `IQ3_K` are on par or slightly better than i-quants with similar size, so before improving these you have to find a way to apply the methods of your PR to `IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S` (one of your TODO items).
* On `TQ2_0` being faster than `IQ1_S`: in theory, sure. In practice, the table below shows what I observe with the PR branch for `TQ2_0`, and with `ik_llama.cpp` for `IQ1_S` (using the row-interleaved variant `IQ1_S_R4`): 

 | model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| llama 8B TQ2_0 - 2.06 bpw ternary |   2.72 GiB |     8.03 B | CPU        |      16 |         pp512 |        153.05 ± 0.29 |
| llama 8B TQ2_0 - 2.06 bpw ternary |   2.72 GiB |     8.03 B | CPU        |      16 |         tg128 |         23.79 ± 0.00 |
| llama 8B IQ1_S_R4 - 1.5 bpw    |   2.39 GiB |     8.03 B | CPU        |      16 |         pp512 |    184.46 ± 1.36 |
| llama 8B IQ1_S_R4 - 1.5 bpw    |   2.39 GiB |     8.03 B | CPU        |      16 |         tg128 |     26.86 ± 0.00 |


### @jukofyork's ideas 

If you start with a fully symmetric probability distribution (not always the case, but for simplicity let's assume it is fully symmetric), and you draw a **finite** number of random samples from it (the wights in one quantization block), you then scale the sampled values such that the maximum magnitude value **always takes the same scaled value**, you end up with a non-symmetric probability distribution for the **scaled samples**. The smaller the sample size, the larger the asymmetry. With the sample size approaching infinity, the observed probability distribution will become symmetric. You can ask WolframAlpha about it, or you can write a simple script that samples 32 values from a Gaussian distribution, scales, and scores the resulting scaled pdf.

Anyway, this is why the `IQ4_NL` (and `IQ4_XS`, as well as the `IQ2_K, IQ3_K` quants from this repository) quant lookup tables are asymmetric (and not because I'm a moron who didn't know how to make a symmetric function). But, if you don't accept this for granted (you most likely don't), just go and replace `kvalues_iq4nl` in `ggml-quants.c` with your symmetric variant, and watch the disaster that ensues. You need to do it at a few more places because for some reason this table is not in `ggml-common.h` as it should be. 

___
<sup>1</sup> I know, I know. The Internet Gods have spoken: PPL doesn't tell us anything and is completely useless; KLD is the one and only one true measure of quantization quality. But me, not being a religious person, and having quite a bit of research experience under my belt,  I don't take the God's opinions for granted. I have written elsewhere about the equivalence of PPL and KLD for an infinitely large test corpus, and about the superiority of PPL for a test corpus of limited size, so I will not repeat myself here.

---

#### 🗣️ Discussion

👤 **jukofyork** replied the **2025-03-25** at **12:48:44**:<br>

> @compilade has submitted an [interesting PR](https://github.com/ggml-org/llama.cpp/pull/12557) in the mainline `llama.cpp` repository. As it is often the case, @jukofyork has improvement ideas. As both pinged me, and as I no longer hang around in the `llama.cpp` project, I'll address the pings here.

> ### @jukofyork's ideas
> 
> If you start with a fully symmetric probability distribution (not always the case, but for simplicity let's assume it is fully symmetric), and you draw a **finite** number of random samples from it (the wights in one quantization block), you then scale the sampled values such that the maximum magnitude value **always takes the same scaled value**, you end up with a non-symmetric probability distribution for the **scaled samples**. The smaller the sample size, the larger the asymmetry. With the sample size approaching infinity, the observed probability distribution will become symmetric. You can ask WolframAlpha about it, or you can write a simple script that samples 32 values from a Gaussian distribution, scales, and scores the resulting scaled pdf.
> 
> Anyway, this is why the `IQ4_NL` (and `IQ4_XS`, as well as the `IQ2_K, IQ3_K` quants from this repository) quant lookup tables are asymmetric (and not because I'm a moron who didn't know how to make a symmetric function). But, if you don't accept this for granted (you most likely don't), just go and replace `kvalues_iq4nl` in `ggml-quants.c` with your symmetric variant, and watch the disaster that ensues. You need to do it at a few more places because for some reason this table is not in `ggml-common.h` as it should be.

Just to be clear: I wasn't implying you had done anything wrong and merely showing something that I had noticed and spent a couple of hours playing with last year (which I never mentioned before as it wasn't clear it was of any use nor related to anything useful).

I'm sorry if I've come across badly as this isn't my intention - I've nothing to gain from any of this, but just find it interesting :) If you search my nick you can find similar posts by me on the now dead 2+2 forums (everything is on discord now sadly) on similar topics from 25+ years ago!

---

👤 **ikawrakow** replied the **2025-03-25** at **14:28:09**:<br>

@jukofyork Sorry if I have come across a bit harsh. But it is interesting stuff indeed, so we all can get passionate about it.

Anyway, attached is a very simple C++ program that illustrates the asymmetry of the scaled distribution. Here is what it does:
* It picks $N$ random points, either uniformly in $[-1,1]$ or from a Gaussian distribution with $\sigma = 1$ (command line argument)
* It finds the minimum and maximum values in the sample $x_{\rm min}$ and $x_{\rm max}$
* It determines a scale such that the value with the larger absolute value is at -1. I.e., if $|x_{\rm min}| > |x_{\rm max}|$, then $s = -1/x_{\rm min}$, else $s = -1/x_{\rm max}$. It than takes the other extremum (the one with the lower absolute value), and computes $x_s = s x_{\rm other}$.
* It repeats the above $M$ times and computes the average of the observed $x_s$

Here is a plot of the computed average as a function of sample size $N$. For a sample of just 2 points, the average is effectively zero. If the distribution of scaled values was symmetric, the average should be 1 (or very close to 1). We see that this is not the case. For a Gaussian distribution we are quite far away from the symmetric value of 1 that we expect for $N \to \infty$ even for $N = 32$ (the typical block size used in many k- and i-quants). I have used
```
g++ -O3 distr1.cpp
./a.out 1000 -32 >test1.out
./a.out 1000 -32 1 > test2.out
```
to generate the data in the graph (a negative sample size will cause the program to loop between 2 and the absolute value of the argument given).

![distr](https://github.com/user-attachments/assets/81286fac-86ec-4f20-873e-24d6eb18f36c)

[distr1.cpp.gz](https://github.com/user-attachments/files/19449673/distr1.cpp.gz)

---

👤 **ikawrakow** replied the **2025-03-25** at **15:01:41**:<br>

Here is another very simple C++ program:
* Pick $N$ random values
* Sort them in increasing order. Let's the sorted values be $x_i$
* If $|x_0| > |x_{N-1}|$, then $s = -1/x_0,\quad\tilde{x}_i = s x_i$
* Else $s = -1/x_{N-1}$ and $\tilde{x}_i = s x_{N-1-i}$ (don't know why it doesn't show the equation correctly)
* Compute the average of the scaled $\tilde{x}_i$ over a given number of samples.

With this, we get this graph. It looks very similar to what one gets by doing an actual block-wise quantization with non uniform values.
![distr2](https://github.com/user-attachments/assets/92a9e89c-297b-4a1c-be36-675499e094c5)

[distr2.cpp.gz](https://github.com/user-attachments/files/19450493/distr2.cpp.gz)

---

👤 **compilade** replied the **2025-03-25** at **16:25:49**:<br>

@ikawrakow

> First of all, this is a nice piece of work, so congratulations!

Thank you. Your existing work on `imatrix` definitely made it easier to try this kind of weighted rounding algorithms on actual models. At first the idea only applied to ternarization with no ability to weigh the error: <https://github.com/microsoft/BitNet/discussions/112>.

> Without imatrix, the existing quantization methods are strictly better than your PR as measured by perplexity

Right. I will consider reverting back to the existing quantization methods when `imatrix` is not used (although for `Q3_K`, I still think `make_q3_quants` has some problems when the sign of the absmax value is positive (according to the equirectangular projections, in that case it looks like almost exactly like what `Q3_0` would (in the upper left part)), which could be fixed).

I was hoping the more exhaustive algorithms would always be better (since they *are* better at minimizing the weighted squared error), but when they optimize the wrong thing (when no `imatrix` is given) can be worse, except apparently for some models like `Qwen2.5-Coder-3B-Instruct`. 

But I also suspect the default weights for the weighted rounding without `imatrix` could be improved (but at that point I guess I should only change what rounding algorithm is used *if* I find those better default weights (which I thought I did from the results of `Qwen2.5-Coder-3B-Instruct`, but apparently not in general)).

Aside: *is there* a generally better solution for the default importance weights (without `imatrix`)? (It seems the heuristics between quant types disagree: some use `x[i] * x[i]`, others `fabsf(x[i])`, and others `sqrtf(sum_x2/N) + fabsf(x[i])` (Note that I did read <https://github.com/ikawrakow/ik_llama.cpp/discussions/140>, I'm not questioning that these were better in practice in their respective cases))
I think this depends on the weighted rounding algorithm with which the weights are used (since the behaviors can be different).

> `IQ4_NL` is hit-or-miss - sometimes slightly better, sometimes slightly worse, but overall not much of a difference apart from the 5X increase in quantization time

Strange, the increase in quantization time for `IQ4_NL`  with `imatrix` is only slightly more than 2× for me, and close to none (1×) when no `imatrix` is provided. There is room for improvement in the performance of `make_qkxh_nl_quants` because I did not yet extensively profile it with `perf` except for a previously slower `qsort`-based version (which *really was* 5× slower).

And there are still some adjustments I did not try yet and which could improve both the time (by a noticeable factor) and perplexity (hopefully), which is to add the same "clamping protection" as my linear weighted rounding algorithms (e.g. in `make_qkxh_quants`, the inverse scales which would clamp the `x[i]` with the biggest `w[i] * fabsf(x[i])` are not tried (since this *did* improve the PPL and KLD with `imatrix` for linear quants like `Q3_K`, `Q4_0` and `Q5_0`)). But it might also not help in which case I'm considering reverting to the existing `IQ4_NL` quantization algorithm, even though it makes less satisfying equirectangular projections.

I value your feedback, which is why I'll try to improve on this point (or exclude the changes to `IQ4_NL`).

> You do use `Q8_0` for the output tensor, I think it would be better to also use `Q8_0` for token embeddings when using `--pure`.

I do use `Q8_0` for the token embeddings too in my tests. The example command I've included in the PR description **does** specify `--token-embedding-type q8_0`

```console
$ ./bin/llama-quantize --imatrix <some-file.imatrix> --token-embedding-type q8_0 --output-tensor-type q8_0 --pure <source.gguf> <quant.gguf> <quant-type>
```

> RMSE (or cosine similarity) are just surrogates, so finding a better solution does not automatically lead to better quantization quality.

Yeah, I did notice that. The search algorithms I've made can be adapted to other metrics (although that can also be said of the existing algorithms for k-quants, since they also use weighted squared error), as long as they can be calculated cumulatively.

I'd like to find better surrogates, and more exhaustive search algorithms which are not brute-force (yet still yield optimal-looking results) can help with that, even though for now minimizing weighted squared error on the model tensors doesn't quite match the actual thing we want to minimize (PPL and KLD), which makes your carefully tuned heuristics superior for now.

> Case in point, Q3_K_M with your PR often has a higher PPL than the existing quantization, despite being clearly better with `--pure`

On which model(s) did you observe this? I'd like to reproduce this observation.

> I have written elsewhere about the equivalence of PPL and KLD for an infinitely large test corpus, and about the superiority of PPL for a test corpus of limited size, so I will not repeat myself here.

Right, but the test corpus is not infinite, and for a small test corpus I actually find KLD faster for meaningful comparisions (because the ± error goes down faster than for `ln(PPL(Q)/PPL(base))`, and so sometimes when I'm not using a GPU I don't have to leave it running that long to know if a change is meaningful when tweaking some things).

But I agree PPL is more convenient for quickly comparing versions of quants of a lot of different models (because the logits files get big really fast), at least when using a GPU.

> But it is interesting stuff indeed, so we all can get passionate about it.

Yes, totally agree! And technically I already got what I wanted out of these algorithms (even if they are not merged or not better), which is the very nice plots they can make to hopefully help me understand a bit more the representable vector space of both linear and non-linear quants, especially when viewed appropriately in a 360 degree panorama viewer: <https://blobs.compilade.net/pannellum.htm#panorama=equirectangular-iq4nl-qkxs-2048.png>.

---

👤 **ikawrakow** replied the **2025-03-25** at **16:53:43**:<br>

> Aside: is there a generally better solution for the default importance weights (without imatrix)? (It seems the heuristics between quant types disagree: some use x[i] * x[i], others fabsf(x[i]), and others sqrtf(sum_x2/N) + fabsf(x[I])

It is a heuristic. Trial and error. IIRC, higher bpw quants do better with a stronger large magnitude weighting (e.g., $x^2$), with lower bpw $|x|$ or similar is generally better.

 > On which model(s) did you observe this? I'd like to reproduce this observation.

Go back to the basics. Start with LLaMA-v1-7B. I know, nobody uses that today. But then again, almost all of k-quants development was based on the experience with the LLaMA-v1 models, and k-quants have done surprisingly well in the almost two years since they were released on the thousands of models they have been tried on. Even today when I want to try a new quantization idea, I always check performance with LLaMA-v1, LLaMA-v2, and Mistral-7B. Your `IQ4_NL` doesn't do very well on LLaMA-v1-7B - without an imatrix it arrives at a PPL higher than `Q4_0`.

> Strange, the increase in quantization time for IQ4_NL with imatrix is only slightly more than 2× for me,

Oh, I used `ik_llama.cpp` to compare. It is possible that has become much faster than mainline (I haven't used mainline for quite some time). I started testing with DeepSeek-Lite, and almost gave up (your `IQ4_NL` quantization took 302.5 seconds with imatrix). `ik_llama.cpp` does it in 54.5 seconds.

> 👤 **bartowski1182** replied the **2025-03-26** at **17:42:29**:<br>
> Re: quantization speed
> 
> Do you have any loose thoughts on where your crazy speedup may be coming from? Not asking you to do a thorough investigation, but curious if you have an initial place to point me
> 
> 👤 **ikawrakow** replied the **2025-03-26** at **18:16:32**:<br>
> IIRC:
> At some point I was annoyed by the slow quantization speed of quantization types with non-linear grids (`IQ4_XS, IQ4_NL` in mainline, here also `IQ2_KS, IQ2_K, IQ3_K, IQ4_K, IQ5_K, IQ6_K`). The major bottleneck turned out to be finding the bin in which a value falls after scaling. E.g., [this function](https://github.com/ggml-org/llama.cpp/blob/2447ad8a981253a2b8e9f4b31cc8e7fdff83423e/ggml/src/ggml-quants.c#L4562) in mainline, which does a binary search to find the bin. So, I replaced that with functions such as [this one](https://github.com/ikawrakow/ik_llama.cpp/blob/a22250df93fd833a6cb7f310b159ad1b54e4d582/ggml/src/ggml-quants.c#L14528). I think that was the major part. I don't remember if I did additional optimizations and what they were, if any. I would have to go through the old PRs to find out.
> 
> 👤 **compilade** replied the **2025-03-26** at **18:24:02**:<br>
> @bartowski1182
> 
> (EDIT: sorry, I did not see ikawrakow's answer before commenting)
> 
> My guess would be that `best_index_iq4nl` is faster than `best_index_int8`:
> 
> <https://github.com/ikawrakow/ik_llama.cpp/blob/a22250df93fd833a6cb7f310b159ad1b54e4d582/ggml/src/ggml-quants.c#L14518-L14533>
> 
> And `best_index_int8` does lots of comparisons instead of using a lookup table more directly (doesn't seem to render inline since it's from a different repo (mainline `llama.cpp`)):
> 
> <https://github.com/ggml-org/llama.cpp/blob/2447ad8a981253a2b8e9f4b31cc8e7fdff83423e/ggml/src/ggml-quants.c#L4562-L4571>
> 
> I will check if (and how) `best_index_iq4nl` affects the equirectangular projection of `IQ4_NL`, since that seems relevant.
> (EDIT: it doesn't seem to change anything at a cursory glance. So it is pretty much equivalent.)
> 
> 👤 **ikawrakow** replied the **2025-03-26** at **18:40:39**:<br>
> Here some napkin math: @compilade said that their approach is only 2X slower than the master branch in mainline. If I use the DeepSeek-Lint values, it means mainline will quantize it in 150 seconds instead of 300 seconds. If you add this optimization, it will become 50 seconds (using round values to make it easier to follow). You then add 150 seconds for the heap search, and it becomes 200 seconds. So, 4X slower than `ik_llama.cpp`, but only ~30% slower than the current state of mainline.
> 
> 👤 **compilade** replied the **2025-03-26** at **19:26:28**:<br>
> @ikawrakow My implementation (with the cumulative search) unfortunately cannot use this optimization, because it doesn't use `best_index_int8` anyway. The reason my implementation is slow is because it's too exhaustive. It calculates `sumqx` and `sumq2` for *all* scales which would result in a distinct quantization, and it tests both signs. That is `(32*(7+8))+1 = 481` distinct scales compared per block of 32, compared to the `(2*7+1)+1 = 16` scales compared by the implementations which use either `best_index_int8` or `best_index_iq4nl`.
> 
> It's nice that it's not `481/16 = 30` times slower, though 6× does seem too slow, I agree.
> 
> The only ways to make the cumulative search faster is to reduce how many scales it searches (which for linear quants is easier because more of them are equivalent and can be skipped), or to make the cumulative step faster.
> 
> (It might be possible to mix both approaches to search for more than 16 scales at 1× speed (or faster))
> 
> 👤 **bartowski1182** replied the **2025-03-26** at **19:35:38**:<br>
> Appreciate the insights, thanks!

---

👤 **ikawrakow** replied the **2025-03-28** at **09:36:09**:<br>

@compilade @bartowski1182 

You may be interested in PR #295

---

👤 **ubergarm** replied the **2025-03-29** at **17:57:59**:<br>

While not directly related to the quants specific to #295 , I did just release what may be one of the best quants (for generation quality) in its size class for `V3-0324` on huggingface [ubergarm/DeepSeek-V3-0324-GGUF](https://huggingface.co/ubergarm/DeepSeek-V3-0324-GGUF) cooking with `ik_llama.cpp`. It also still fits 32k context in under 24GB VRAM and can hit over 4 tok/sec tg mmap'ing on my 9950x 96GB + 3090TI 24GB VRAM rig using `-ser 6,1` sacrificing minimal perplexity.

It only works with `ik_llama.cpp` as even with experimental mainline PRs [fairydreaming:deepseek2-mla-exp](https://github.com/ggml-org/llama.cpp/pull/11446) and [sl/custom-tensor-offload](https://github.com/ggml-org/llama.cpp/pull/11397) you still need support for `IQ3_K_R4`/`IQ2_K_R4` which is only available here.

I haven't done full perplexity and benchmarking comparisons across the major quant cookers versions, but have a rough table showing the differences between ubergarm, @bartowski1182, @danielhanchen (unsloth), and eventually mradermacher's recipes. I'll add it in the fold here for convenience.

Big thanks to y'all doing so much inspirational work and making this stuff more and more accessible!

:point_down: 
<details>

<summary>:point_left: V3-0324 quant recipe comparison table</summary>

| | [ubergarm/DeepSeek-V3-0324-IQ2_K_R4](https://huggingface.co/ubergarm/DeepSeek-V3-0324-GGUF?show_file_info=DeepSeek-V3-0324-IQ2_K_R4%2FDeepSeek-V3-0324-IQ2_K_R4-00001-of-00005.gguf) | [bartowski/DeepSeek-V3-0324-Q2_K_L](https://huggingface.co/bartowski/deepseek-ai_DeepSeek-V3-0324-GGUF?show_file_info=deepseek-ai_DeepSeek-V3-0324-Q2_K_L%2Fdeepseek-ai_DeepSeek-V3-0324-Q2_K_L-00001-of-00007.gguf) | [unsloth/DeepSeek-V3-0324-UD-Q2_K_XL](https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF?show_file_info=UD-Q2_K_XL%2FDeepSeek-V3-0324-UD-Q2_K_XL-00001-of-00006.gguf) | [mradermacher/DeepSeek-V3-0324-i1-GGUF-Q2_K](https://huggingface.co/mradermacher/DeepSeek-V3-0324-i1-GGUF) |
| --- | --- | --- | --- | --- |
| **Overview**                       |            |        |        |        |
| `tensor_count`                     |   267      |   190  |   253  |        |
| `kv_count`                         |    53      |    53  |    49  |        |
| `split.tensors.count`              |  1147      |  1025  |  1025  |        |
| `token_embd.weight`                | `Q8_0`     | `Q8_0` | `Q4_K` |        |
| File Size (GiB)                    |   227      |   228  |   231  |        |
| **Multi-Head Latent Attention**    |            |        |        |        |
| `blk.*.attn_kv_b.weight`           | `Q8_0`     |   n/a  |   n/a  |   n/a  |
| `blk.*.attn_k_b.weight`            | `Q8_0`     |   n/a  |   n/a  |   n/a  |
| `blk.*.attn_v_b.weight`            | `Q8_0`     |   n/a  |   n/a  |   n/a  |
| **Dense Layers**                   |            |        |        |        |
| `blk.[0-2].attn_kv_a_mqa.weight`   | `Q8_0`     | `Q2_K` | `Q6_K` |        |
| `blk.[0-2].attn_kv_a_norm.weight`  | `F32`      |  `F32` |  `F32` |        |
| `blk.[0-2].attn_kv_b.weight`       | `Q8_0`     | `Q2_K` | `Q6_K` |        |
| `blk.[0-2].attn_norm.weight`       | `F32`      |  `F32` |  `F32` |        |
| `blk.[0-2].attn_q_a.weight`        | `Q8_0`     | `Q2_K` | `Q4_K` |        |
| `blk.[0-2].attn_q_a_norm.weight`   | `F32`      |  `F32` |  `F32` |        |
| `blk.[0-2].attn_q_b.weight`        | `Q8_0`     | `Q2_K` | `Q4_K` |        |
| `blk.[0-2].ffn_down.weight`        | `Q8_0`     | `Q3_K` | `Q6_K` |        |
| `blk.[0-2].ffn_gate.weight`        | `Q8_0`     | `Q2_K` | `Q4_K` |        |
| `blk.[0-2].ffn_norm.weight`        | `F32`      |  `F32` |  `F32` |        |
| `blk.[0-2].ffn_up.weight`          | `Q8_0`     | `Q2_K` | `Q4_K` |        |
| `blk.[0-2].attn_output.weight`     | `Q8_0`     | `Q3_K` | `Q4_K` |        |
| **Shared & Routed MoE Layers**     |            |        |        |        |
| `blk.[3-60].attn_kv_a_mqa.weight`  | `Q8_0`     | `Q2_K` | `Q6_K` |        |
| `blk.[3-60].attn_kv_a_norm.weight` | `F32`      | `F32`  | `F32`  |        |
| `blk.[3-60].attn_kv_b.weight`      | `Q8_0`     | `Q2_K` | `Q6_K` |        |
| `blk.[3-60].attn_norm.weight`      | `F32`      | `F32`  | `F32`  |        |
| `blk.[3-60].attn_q_a.weight`       | `Q8_0`     | `Q2_K` | `Q4_K` |        |
| `blk.[3-60].attn_q_a_norm.weight`  | `F32`      | `F32`  | `F32`  |        |
| `blk.[3-60].attn_q_b.weight`       | `Q8_0`     | `Q2_K` | `Q4_K` |        |
| `blk.[3-60].exp_probs_b.bias`      | `F32`      | `F32`  | `F32`  |        |
| `blk.[3-60].ffn_down_exps.weight`  | `IQ3_K_R4` | `Q3_K` | `Q3_K` |        |
| `blk.[3-60].ffn_down_shexp.weight` | `Q8_0`     | `Q3_K` | `Q6_K` |        |
| `blk.[3-60].ffn_gate_exps.weight`  | `IQ2_K_R4` | `Q2_K` | `Q2_K` |        |
| `blk.[3-60].ffn_gate_inp.weight`   | `F32`      | `F32`  | `F32`  |        |
| `blk.[3-60].ffn_gate_shexp.weight` | `Q8_0`     | `Q2_K` | `Q4_K` |        |
| `blk.[3-60].ffn_norm.weight`       | `F32`      | `F32`  | `F32`  |        |
| `blk.[3-60].ffn_up_exps.weight`    | `IQ2_K_R4` | `Q2_K` | `Q2_K` |        |
| `blk.[3-60].ffn_up_shexp.weight`   | `Q8_0`     | `Q2_K` | `Q4_K` |        |
| `blk.[3-60].attn_output.weight`    | `Q8_0`     | `Q3_K` | `Q4_K` |        |
| **Important Matrix & Perplexity**  |            |        |        |        |
| `imatrix.dataset`                  | `calibration_data_v5_rc.txt`| `calibration_datav3.txt` | n/a | ? |
| Final PPL (wiki.test.raw)          | 3.5614 +/- 0.02001  | ?      | ?  | ? |


</details>

:point_up:

> 👤 **ikawrakow** replied the **2025-03-29** at **18:18:55**:<br>
> I would be really curious to see the PPL values of the other quant cookers.
> 
> 👤 **bartowski1182** replied the **2025-03-29** at **18:42:51**:<br>
> How many chunks of wiki test raw are you using for PPL? If you give your exact command I can get you the PPL for my own quant
> 
> It's very intriguing. I know that most likely the unsloth one will be better than my own since he went out of his way to optimize the tensor types for that model which is just not something I have the throughput to handle 😅
> 
> Also don't really want to make the same ones as him and release them since it would just be ripping off his work 🤷‍♂️
> 
> Interesting stuff overall though
> 
> 👤 **ubergarm** replied the **2025-03-29** at **19:06:34**:<br>
> Yeah I'm curious too! Bartowski you do use imatrix though, which I don't think unsloth does. So  so not sure how that would make up for the smaller tensor types.
> 
> I just ran the `Q8_0` for baseline comparison and got this result:
> 
> >Final estimate: PPL = 3.2454 +/- 0.01773
> 
> Here is the methodology including exact wiki.text.raw and commands:
> 
> <details>
> 
> <summary>:point_right: Details and Methodology :point_left: </summary>
> 
> ```bash
> $ cd ik_llama.cpp
> $ git rev-parse --short HEAD
> 4819257c
> 
> $ wget https://github.com/user-attachments/files/19090237/wiki.test.raw.gz
> $ gunzip wiki.test.raw.gz
> $ sha256sum wiki.test.raw
> 173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08  wiki.test.raw
> 
> # CPU+GPU Perplexity Run
> $ CUDA_VISIBLE_DEVICES="0," \
> ./build/bin/llama-perplexity \
>     --model /mnt/raid/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ2_K_R4.gguf \
>     -ctk q8_0 \
>     -mla 2 -fa \
>     -amb 512 \
>     -fmoe \
>     --ctx-size 512 \
>     --ubatch-size 512 \
>     -f wiki.test.raw \
>     --seed 1337 \
>     --n-gpu-layers 63 \
>     --override-tensor exps=CPU \
>     --threads 24
> 
> # CPU only Perplexity Run (for big `Q8_0`)
> $ numactl -N 1 -m 1 \
> ./build/bin/llama-perplexity \
>     --model /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-Q8_0.gguf \
>     -ctk q8_0 \
>     -mla 3 -fa \
>     -amb 512 \
>     -fmoe \
>     --ctx-size 512 \
>     --ubatch-size 512 \
>     -f wiki.test.raw \
>     --seed 1337 \
>     --numa numactl \
>     --threads 128
> 
> llama_print_timings:        load time =    3493.83 ms
> llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
> llama_print_timings: prompt eval time = 4081619.28 ms / 287232 tokens (   14.21 ms per token,    70.37 tokens per second)
> llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
> llama_print_timings:       total time = 4132068.91 ms / 287233 tokens
> 
> Final estimate: PPL = 3.2454 +/- 0.01773
> ```
> 
> </details>
> 
> One other nice thing about `ik_llama.cpp` is you can customize the layers using a script without maintaining a llama.cpp code fork. I included the [script I used on the model card](https://huggingface.co/ubergarm/DeepSeek-V3-0324-GGUF#quantize-script).
> 
> Finally, I'm not sure what imatrix text mradermacher uses to make imatrix, but I did a [quick comparison](https://gist.github.com/tristandruyen/9e207a95c7d75ddf37525d353e00659c?permalink_comment_id=5519433#gistcomment-5519433) of two otherwise identical quantizations using bartowski's imatrix and a slightly updated input text. They give similar perplexity against wiki.text.raw, for whatever that is worth hah...
> 
> Anyway, yeah thanks for all your effort! I dunno how y'all keep up with the torrent of near weekly big model releases lately! Cheers!
> 
> 👤 **ikawrakow** replied the **2025-03-29** at **19:06:35**:<br>
> I think @ubergarm can do the full PPL in less than an hour with their Xeon server. I don't know what kind of hardware you have.
> 
> > ... since he went out of his way to optimize the tensor types for that model 
> > Also don't really want to make the same ones as him and release them since it would just be ripping off his work
> 
> I'm sure you are aware that quantization mixes have been in `llama.cpp` since the release of k-quants. All of those use more bits for the first few `ffn_down` layers. Also all of them use more bits for the attention tensors in MoE models. If you look at the Unsloth's so called "dynamic" quants, it is easy to see that with a small change of the function that determines the quantization type to handle the different names of the DeepSeek tensors (and the presence of shared experts), you will get basically what they used. Did they mention that? Of course not. So now the entire industry knows that Unsloth invented "dynamic" quants.
> 
> 👤 **bartowski1182** replied the **2025-03-29** at **20:14:48**:<br>
> Yeah I did browse through his repo to check the changes he made, I do understand the overall nature of the quantization mixes and his adjustments made, and I know I could either pull his fork or make similar changes of my own to get the same results but just out of principle don't want to rehost if I'm not actually adding anything to the process
> 
> I've got myself an EPYC server so things run pretty okay on my end as well, I'm just lacking on the GPU front for some things :)
> 
> Unsloth also did a weird thing by releasing truly (I think) "dynamic" BnB quants at the same time as "dynamic" DeepSeek GGUF quants, so the naming feels a bit off, but there clearly is some value to be gained by manually altering the decision making for tensor types to favour some over others with DeepSeek, the generic existing one is leaving performance on the table
> 
> Of course I'd like to know if the efforts in this branch more than make up for that, it wouldn't surprise me at all.. 
> 
> > All of those use more bits for the first few ffn_down layers. Also all of them use more bits for the attention tensors in MoE models
> 
> This part however I was not explicitly aware of, but still in terms of raw bits per weight, unsloth's mix seems superior (at least in the tests he has ran, PPL, KLD, and additional tests would be good to see if it's genuinely big improvements or if it's actually similar overall)
> 
> 👤 **saood06** replied the **2025-03-30** at **01:51:10**:<br>
> Since mradermacher doesn't use gguf split you may have to use [gguf-py/scripts/gguf_dump.py](https://github.com/ikawrakow/ik_llama.cpp/blob/main/gguf-py/scripts/gguf_dump.py) to get the metadata.
> 
> > 👇
> > 👈 V3-0324 quant recipe comparison table
> > ☝️
> 
> You can probably remove tensor_count doesn't matter, as it changes based on split size and kv_count also doesn't really mean much it's just the number of entries of metadata from your table.
> 
> 👤 **ikawrakow** replied the **2025-03-30** at **05:44:14**:<br>
> > This part however I was not explicitly aware of, but still in terms of raw bits per weight, unsloth's mix seems superior
> 
> Superior compared to what? To unmaintained `llama.cpp`? Where @compilade's PR 12557 is the first noteworthy thing related to quantization that has happened since I left the project more than a year ago?
> 
> Let's take a look at a few examples. 
> 
> [This line](https://github.com/ggml-org/llama.cpp/blob/af6ae1efb27a9a7c3f7f7f84639d2243f7303ac1/src/llama-quant.cpp#L250) and the following checks if this is an attention tensor, and if we are dealing with a MoE model. It worked for Mixtral8x7B, which was the only serious MoE model at the time. But in DeepSeek the most important attention tensor is `attn_kv_b`, and we are not having exactly 8 experts, so we don't get the intended behavior.
> 
> [This line](https://github.com/ggml-org/llama.cpp/blob/af6ae1efb27a9a7c3f7f7f84639d2243f7303ac1/src/llama-quant.cpp#L316) sets more bits for the attention output tensor. Again, it fails because DeepSeek doesn't have exactly 8 experts, and no-one of the 1000+ `llama.cpp` contributors knew how to adapt it to the MoE models that came out after Mixtral8x7B.
> 
> When the quantization mix strategies for MoE were written, experts were in separate tensors named `blk.X.ffn_up/gate/down.Y.weight` (where `X` was the layer index and `Y` the expert index). Then somebody decided to combine the experts into a single tensor named `blk.X.ffn_up/down/gate_exps.weight`, but did not change the code that decides on the quantization mix. Voila, you have the `QX_K_M` "dynamic" quants not working as intended.
> 
> Take a look at the code block that follows `} else if (name.find("ffn_down") != std::string::npos) {`. Several of the quantization type modifications use more bits for the first `1/8` of the layers. Which is 7 for DeepSeek-V3/R1. In how many layers do Unsloth use more bits for `ffn_down` in their "carefully tuned dynamic" quants?
> 
> 👤 **bartowski1182** replied the **2025-03-30** at **15:33:58**:<br>
> > Superior compared to what? To unmaintained llama.cpp? Where @compilade's PR 12557 is the first noteworthy thing related to quantization that has happened since I left the project more than a year ago?
> 
> I mean yeah I did mention that I wouldn't be surprised if this branch has superior performance over even what he did 🤷‍♂️ I do recognize the stale state llama.cpp has been left in with regards to SOTA quantization performance
> 
> I'm also not attempting to advocate his work or claim it's a God send, I recognize what it is and what it's being compared to
> 
> Against llama.cpp's IQ2_XXS, it seems to perform closer to the original weights in terms of at least behaviour 
> 
> That's not to say it's anywhere near SOTA or even necessarily close to what you've achieved here, just a factual observation to be used as evidence that in llama.cpp there's clearly performance being left on the table
> 
> That's a very interesting observation about the MoE code though containing a quite glaring bug, I wonder how much fixing that alone gets us back.. presumably a lot since as you mentioned most of the changes in the branch were about those early layers.
> 
> I also recognize the fact that since you left quantization itself has definitely gone to the backburner, I'm very thankful to compilade for his efforts but yeah, not quite the same since
> 
> I'm also surprised no one has come around and attempted to upstream some of your changes, several seem like just free performance gains, others are understandably more complex but there's certainly a few low hanging fruit that are just being ignored (and yes I recognize the irony of not doing it myself while complaining others aren't doing it)
> 
> 👤 **ikawrakow** replied the **2025-03-30** at **17:03:32**:<br>
> The only reason I started this discussion was that you wrote above "... it would just be ripping off his work". And the point I was trying to make was that it would be perfectly fine to rip off their work as this is exactly what they did.
> 
> 👤 **bartowski1182** replied the **2025-03-30** at **17:26:34**:<br>
> Oh I mean, fair haha. I guess I meant I don't want to strictly 1:1 copy his repo and release identical quants
> 
> But you're definitely right that his work is basically just a bandage solution that happens to be the proper way to handle MoE models in general
> 
> I do highly appreciate the insight though for the record, I don't mean to come off as argumentative or dismissive! I'll be looking into what you suggested for sure
> 
> 👤 **bartowski1182** replied the **2025-03-30** at **19:24:25**:<br>
> @ikawrakow would you mind if I took inspiration from your changes to https://github.com/ikawrakow/ik_llama.cpp/blob/main/src/llama.cpp for some upstream work on llama_tensor_get_type? "inspiration" in this case would likely mean just straight up copying any changes that, to my untrained eye, seem strictly better and without risk of negatives (since I wouldn't discount the possibility some may be negative without other appropriate changes throughout the system)
> 
> 👤 **ikawrakow** replied the **2025-03-31** at **06:01:25**:<br>
> Sure, go ahead. I see I haven't actually changed all occurrences of `n_expert == 8` to `n_expert >= 8`, so you may want find/replace all when making the change.
> 
> Here people now use custom rules for making quants, so you may want to explore this as well. If you stick to quants available in mainline `llama.cpp`, you can "cook" the quants you publish with `ik_llama.cpp`.
> 
> 👤 **bartowski1182** replied the **2025-04-01** at **23:20:00**:<br>
> @ubergarm I finished PPL of my original Q2_K upload and a new one I've added with changes from here and also just copying a bit of other work in the area
> 
> llama.cpp main: 3.9012 
> 
> my fork: 3.6868 
> 
> considering the size only increased by 1%, i'm pretty stoked with that PPL improvement, and while yours is clearly still better, llama.cpp main is missing lots of ikawrakow's magic so it's not bad!
> 
> 👤 **saood06** replied the **2025-04-02** at **00:19:01**:<br>
> > I finished PPL of my original Q2_K upload and a new one I've added with changes from here and also just copying a bit of other work in the area
> > 
> > llama.cpp main: 3.9012
> > 
> > my fork: 3.6868
> > 
> > considering the size only increased by 1%, i'm pretty stoked with that PPL improvement, and while yours is clearly still better, llama.cpp main is missing lots of ikawrakow's magic so it's not bad!
> 
> I'm not ubergarm, but thank you for this, I'm always curious to see PPL numbers and this is interesting.
> 
> 👤 **ubergarm** replied the **2025-04-02** at **19:26:29**:<br>
> > @ubergarm I finished PPL of my original Q2_K upload and a new one I've added with changes from here and also just copying a bit of other work in the area
> > 
> > llama.cpp main: 3.9012
> > 
> > my fork: 3.6868
> > 
> > considering the size only increased by 1%, i'm pretty stoked with that PPL improvement, and while yours is clearly still better, llama.cpp main is missing lots of ikawrakow's magic so it's not bad!
> 
> Hey that is a nice drop in PPL for 1% size increase! Ohh sweet I see your [new Q2_K_L-V2](https://huggingface.co/bartowski/deepseek-ai_DeepSeek-V3-0324-GGUF#v2-uploads) variant! I wouldn't say mine is "better" given removing some weight in the GPU tensors possibly allows yours to run 64k context in under 24GB VRAM ([which mine only fits 32k](https://www.reddit.com/r/LocalLLaMA/comments/1joyl9t/comment/ml1lgob/)).
> 
> Also interesting that [suddenly today mainline llama.cpp merged in `-ot` support!](https://github.com/ggml-org/llama.cpp/pull/11397). Curious what they will do with [MLA support](https://github.com/ggml-org/llama.cpp/pull/11446).
> 
> Cheers!
> 
> 👤 **bartowski1182** replied the **2025-04-03** at **03:10:18**:<br>
> Opened the PR here:
> 
> https://github.com/ggml-org/llama.cpp/pull/12727
> 
> that Q2_K_L-V2 will be replaced with a SLIIIIGHTLY better one probably tomorrow, but it's basically the same overall, just a few small bumps for another couple hundred mb
> 
> 👤 **danielhanchen** replied the **2025-04-03** at **03:41:53**:<br>
> Oh hi! I didn't expect to be tagged - @bartowski1182 you're more than welcome to use the llama.cpp fork I have :)
> 
> @ikawrakow Much apologies if people are mis-representing I "invented" dynamic quants, which is far from the truth. Appreciate the work you do, and keep it up - and ignore all the haters - your code is great!
> 
> @ubergarm Great work on the quant as well! I was planning to do imatrix for all quants from now on, but I'm still trying to get the calibration dataset done specifically for instruct models - reasoning models are also a bit more complex.
> 
> 👤 **danielhanchen** replied the **2025-04-03** at **03:45:49**:<br>
> It was actually pure coincidence on making the dynamic quants for DeepSeek R1, V3, since unfortunately as @ikawrakow mentioned, `llama.cpp` also quantizes the shared experts and dense layers the same as the rest of the model - my changes are at https://github.com/unslothai/llama.cpp/
> 
> But the main motivation for "dynamic quants" was due to bitsandbytes and vLLM for finetuning, not actually llama.cpp as @bartowski1182 mentioned. For eg in Gemma 3, I did both activation and weight error analysis to see which parts to quantize / not quantize:
> ![image](https://github.com/user-attachments/assets/1586b89f-b985-47cb-88f1-26bb5b974087)

---

👤 **saood06** replied the **2025-04-11** at **03:06:19**:<br>

@danielhanchen 

For Maverick you reported hitting this over protectiveness issue in llama.cpp

![image](https://github.com/user-attachments/assets/46f8f974-0e6d-41fd-942b-3e9cbce4475c)

>We tried adding more uncommon languages to our calibration dataset, and tried using more tokens (1 million) vs Scout's 250K tokens for calibration

That issue has been addressed here in #202 but you may need to adjust it to allow 10% missing to get the blk.1 tensors as well (but block 45 is below 50% which seems very odd).