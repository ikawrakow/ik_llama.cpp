### 🔀 [#113](https://github.com/ikawrakow/ik_llama.cpp/pull/113) - Trellis quantization

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-11-15 |
| **Updated** | 2025-06-01 |

---

#### Description

The latest quantization hype is `QTIP` - [paper](https://arxiv.org/pdf/2406.11235), [repository](https://github.com/Cornell-RelaxML/qtip). They use a Trellis approach and report impressive results, so I decided to look into this more closely.

This PR implements what they call "3INST" in their paper. Basically, if we have a seed `seed`, we generate `N` quantized values `q_i` via
```
uint32_t u32;
float16_t * h = reinterpret_cast<float16_t*>(&u32)
for i in 0...N-1
    seed = a * seed + b
    u32 = (mask1 & seed) ^ mask2
    q_i = h[0] + h[1]
end
```
where `a, b, mask1` and `mask2` are suitable constants. This generates values that are (nearly) normally distributed. One uses this to describe a group of `N` quants with a single `L`-bit seed (index). Apart from borrowing the "3INST" algorithm from the QTIP paper, the implementation here has noting else in common with QTIP - there are no Hadamard transforms, and no (tail-biting) [Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm) is utilized during quantization. Instead, in the usual i-  and k-quants style, quants are organized in blocks and super-blocks with suitable block scales, and the search for the best seed during quantization is done via a clustering algorithm.

The PR adds 3 new quantization types:
* `IQ2_KT`: `L=16` bits for groups of `N=8` quants. Block size is 32 with a 4-bit block scale, plus a single float scale per tensor row (the 32 bits added by this scale can be safely neglected for typical tensor row sizes), so we end up using 2.125 btw
* `IQ3_KT`: `L=12` bits for groups of `N=4` quants. Block size is also 32 with a 4-bit block scale, so 3.125 bpw
* `IQ4_KT`: `L=15` bits for groups of `N=4` quants. Blocks of 32 with 8-bit block scales, so 4.0 bpw. 

### Quantization accuracy

This figure shows quantization error `PPL(Q)/PPL(bf16)-1` for LLaMA-3.1-8B-Instruct (context length of 8192 tokens). The blue symbols are k-quants, the black symbols are i-quants, cyan symbols are iqk-quants (not available in mainline `llama.cpp`), and the orange symbols are the Trellis quants added by this PR. We do see a small but noticeable improvement compared to i- and iqk-quants, with about 0.2 fewer bpw required to achieve the same quantization error.   

![il31a](https://github.com/user-attachments/assets/b899bc97-9a5e-40c1-83bd-fd0bbb0023c1)

How does this compare to the QTIP paper? Unfortunately they report results without fine tuning only for LLaMA-v2. The table shows a comparison between the 2-bit quantizations for LLaMA-v2-7B (the QTIP results are taken from Table 3 in their paper, context length is 4096 tokens)

| Quantization | PPL(f16) | PPL (Q) | Quantization error |
|------------: | ----: | ----: | ---: |
| QTIP 2 bpw | 5.12 | 6.82 | 33.2% |
| IQ2_KT | 4.94 | 6.36 | 28.7% |

Although there are small differences between the PPL computed by `llama.cpp` and by the tools used by the QTIP authors, the quantization error as defined above is basically independent of the specifics of the PPL calculation, so we see that the 2 bpw quantization implemented here slightly outperforms QTIP without fine tuning (at the expense of using 0.125 bpw more bits). Given this, and the above graph, my conclusion is that Trellis based quantization is a small improvement compared to i-,k-,iqk-quants, but nowhere near the hype observed around the Internet.

### Performance 

The QTIP authors give TG speed for their 2 bpw variant on an RTX-6000 Ada GPU (see [here](https://github.com/Cornell-RelaxML/qtip?tab=readme-ov-file#fast-inference)) and a 7B LLaMA model. My GPU is RTX-4080 (so same generation as theirs, but lower specs). I did a quick attempt to get QTIP going in my environment to have apples-to-apples performance comparison, but it was not successful, so I will use the ratio between their `f16` performance on the RTX-6000 (55.9 t/s) to my `fp16` performance on the RTX-4080 (46.2 t/s) to translate QTIP performance on the RTX-6000 (188 t/s) to estimated performance on the RTX-4080:
```
QTIP (2 bpw, RTX-4080) = fp16(RTX-4080)/fp16(RTX-6000) * QTIP (2 bpw, RTX-6000) = 46.2/55.9*188 = 155.4 t/s
```
In comparison, I get 194 t/s for `IQ2_KT` (with flash attention enabled, which I assume they also use). These results are with the output tensor left as `f16` (which is what is done in QTIP). `IQ2_XSS` achieves 208 t/s (output as `f16`) or 216 t/s (output as `Q5_K`), so QTIP performance is far behind the performance of a model of similar size using a more efficient quantization.

### Caveats

* Quantization is only implemented for a CPU with `AVX2` support. The search for the optimum seed is extremely expensive (the QTIP authors say "prohibitive" for `L >= 12` without their tail-biting search space reduction), so I had to SIMDify to not have to wait forever for a quantization to finish. This PR being mostly a POC for now, I did not want to spend the time implementing for other instruction sets (or even porting to run on a GPU).
* Even with `AVX2`, quantization is slow - depending on quantization type it takes between 2.5 and 4.5 minutes to quantize LLaMA-3.1-8B on a 32-core Ryzen-5975WX CPU.
* Inference is only implemented on CUDA. Due to the "3INST" algorithm, I expect low performance on the CPU and on the Apple GPU, so did not bother to implement for those.
* There are no quantized matrix-vector kernels, so implementation is via the `DMMV` mechanism in `llama.cpp`. The algorithm outputs float values, so one needs to convert to `int8_t` to use the usual quantized dot products. The cost of this  conversion is likely to (more than) offset any advantage one might gain by using SIMD `int8_t` dot products.

---

#### 💬 Conversation

👤 **saood06** commented the **2025-04-07** at **03:27:46**:<br>

Turboderp was also inspired by QTIP when redoing quantization for their new inference engine found [here](https://github.com/turboderp-org/exllamav3).

There is  graphs and more details showing performance of their quants [here](https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md).

I'm interested and will look into it (maybe when the inference engine matures a bit) as I haven't tested using just my 3090 for a 70B model in a long while (the few recent times I wanted to use a 70B I use quants that are too big to fit my 3090 and thus need to be only partially offloaded).

---

👤 **compilade** commented the **2025-04-07** at **12:17:42**:<br>

> There is graphs and more details showing performance of their quants [here](https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md).

Note that [they did not quantize the embeddings with EXL3](https://old.reddit.com/comments/1jt08di/comment/mlse6qg), while they might have with GGUF (not sure, still needs verification), and this might affect the perplexity graphs since they did not include the size of that tensor in the graphs.

(But since they also untie tied embeddings (to quantize the output tensor), it might be hard to compare fairly depending on the model architecture)

Still looks very promising, though!

---

👤 **saood06** commented the **2025-04-07** at **12:43:17**:<br>

> > There is graphs and more details showing performance of their quants [here](https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md).
> 
> Note that [they did not quantize the embeddings with EXL3](https://old.reddit.com/comments/1jt08di/comment/mlse6qg), while they might have with GGUF (not sure, still needs verification), and this might affect the perplexity graphs since they did not include the size of that tensor in the graphs.
> 
> (But since they also untie tied embeddings (to quantize the output tensor), it might be hard to compare fairly depending on the model architecture)
> 
> Still looks very promising, though!

The linked doc page says "Accounting for quantization of the output layer can make a huge difference in practice, especially for smaller models. So I am including two versions of each perplexity graph, one with bitrate on the horizontal axis, and one that measures the entire VRAM footprint of the weights (not counting the embedding layer which for most inference tasks can be relegated to system RAM.)"

So the bpw chart includes the embeddings layer it seems, and the VRAM one does not (both of which useful so I'm glad they offered both).

>Still looks very promising, though!

Yes.

---

👤 **saood06** commented the **2025-04-07** at **13:25:24**:<br>

> I don't like these plots too much. The y-axis needs to be logarithmic, and it needs to be difference to unquantized, not absolute values (else we are chasing differences between possibly different ways of computing perplexity). Also, they massively overemphasize the low bpw range. If you plot on a log scale, you get a more realistic picture. 

Yes but they are good enough for just looking at a VRAM amount and seeing the expected quality for it with the different quants.

>Either way, yes, trellis quantization can bring a 0.1-0.2 bpw reduction in quantized size for the same model quality.

It is more for exllamaV2 to V3 since EXL2 were much worse at low bpw than i-quants. (People did say it did offered better KV cache due to the Hadamard transform added [here](https://github.com/turboderp-org/exllamav2/commit/324404ebe4e3c4dd0447ffc1290c312de1df02be) than llama.cpp even if the model quantization was not as good).

Even though the performance on ik_llama.cpp is lower for CUDA I still prefer it to exllamaV2 because of iqk quants (and also the side benefit of one API implementation) when running models that fit solely on my 3090.

>But is there any indication of performance? I could get my implementation here to be reasonably performant on CUDA, but expect the CPU implementation to be a disaster performance wise.

Exllama is designed for GPUs (and right now only CUDA with ROCm planned) and they are previewing this alongside a new version of their inference software.

The Readme says, 

"Aside from lifting a few of the most successful features from V2 (such as the generator), ExLlamaV3 is largely rewritten from scratch to provide a cleaner, more modular framework for supporting newer architectures. It also introduces a new SOTA quantization format based on [QTIP](https://github.com/Cornell-RelaxML/qtip)"

"The framework is not yet fully optimized. Performance is lacking, especially on Ampere [...]"

>but expect the CPU implementation to be a disaster performance wise.

That is unfortunate.

---

👤 **saood06** commented the **2025-04-08** at **07:21:43**:<br>

Also I forgot to mention it but I did mention your PR to the QTIP authors shortly after you made this draft PR. They said "It seems like they didn't bother making the weights Gaussian first (the IP part of QTIP) before quantizing with a Gaussian codebook (3INST)."

You say in the PR "This generates values that are (nearly) normally distributed." and in a commit message "I also notices that the 3INST generator is not actually  generating a Gaussian distribution." do you think if you followed the author's suggestion it would result in a meaningful difference in quality or is that something you would expect to not matter as much? (I'm not asking you to implement it if you don't know, I know this PR took a long time, and the fact that it is not CPU friendly means it has limited utility for this repo).

---

👤 **ikawrakow** commented the **2025-04-08** at **07:38:55**:<br>

It depends on what the QTIP authors mean by "they didn't bother making the weights Gaussian first". If they mean that I did not apply a Hadamard transform first, I did try that (QuIP/QuIP#/QTIP they all insist on applying Hadamard transforms to model weights before quantization), but it did not improve the result in any way. The thing about Hadamard transforms and imatrix is that they do not mix well - one needs a special imatrix for that. But I have also tried this, without much success. If they mean that I have missed something in the 3INST implementation, and hence the generated sequence is not normally distributed, and it would be better otherwise, I cannot confirm that either. I did a lot of Monte Carlo stuff in the past, so I know a thing or two about random number sequences. I tried an implementation that produces a perfect Gaussian distribution (and quite a bit more efficiently than theirs), but that made results worse.

I was planning to try a sequence that generates quantized values, so CPU inference will be more efficient. But than I started doing other stuff, so that never materialized.

But do the QTIP authors believe theirs is much better than what I have done? My impression was that it was about the same, give or take.

---

👤 **saood06** commented the **2025-04-08** at **08:02:15**:<br>

> I was planning to try a sequence that generates quantized values, so CPU inference will be more efficient. But than I started doing other stuff, so that never materialized.

That sounds interesting.

>It depends on what the QTIP authors mean by ...
>...
>But do the QTIP authors believe theirs is much better than what I have done? My impression was that it was about the same, give or take.

I don't know, the one line I quoted ("It seems ...") is the only thing they said to me. I was merely asking out of my own curiosity, I have no intention of testing their implementation but I may end up testing the EXL3 implementation once it has matured.

---

👤 **louiehelm** commented the **2025-04-17** at **20:00:44**:<br>

The Hadamard Bros and other people fixated on rotations aren't doing it primarily to improve LLM weight quantization. It's for eliminating downstream outliers in run-time activations + KV-cache so they can successfully quantize those more aggressively down to 4-bits without scrambling model fidelity.

Activations and KV-cache are only more sensitive to quantization because of 5-10 tokens per model that represent attention sinks (like [BOS] or "\n") which typically have activation values >100,000x than all the other tokens. This is why even though 4-bit activations only cause ~0.0001% average error, it still breaks most models because the error is all concentrated in these 5-10 essential tokens. This can cause models to glitch out or loop when they're over-quantized. Activation values for attention sinks (outlier tokens) end up very finely-calibrated during training so most models immediately become flakey when they're perturbed.

There's another way to resolve this besides submitting to the Hadamard cult. [PrefixQuant](https://arxiv.org/abs/2410.05265) is a fairly small patch to KV-cache and activation handling that marks the 5-10 largest outlier tokens and just always pre-caches them into KV-cache in full f32. Then 4-bit quantize all the other activations and kv-cache for huge speed and memory benefits and no quality trade-off.

---

👤 **saood06** commented the **2025-04-18** at **23:11:20**:<br>

> There's another way to resolve this besides submitting to the Hadamard cult.

The author of ExllamaV3 reported that they will attempt other ideas as well and only go back to Hadamard if they don't work better.

---

👤 **saood06** commented the **2025-04-19** at **11:07:35**:<br>

> [PrefixQuant](https://arxiv.org/abs/2410.05265)

Finally got a chance to read the paper.

>is a fairly small patch

Look at "Table 5: Ablation study on quantization techniques used in PrefixQuant" and "Appendix D. More Ablation Results", the blockwise finetune that took 17 hours on Llama-3-70B with an NVIDIA-A100-80GB GPU and it having to be the correct dataset and having all the training parameters exact which contributed to their results. 

>KV-cache and activation handling that marks the 5-10 largest outlier tokens and just always pre-caches them into KV-cache in full f32.

This still sounds useful they reported this took 13 minutes on Llama-3-70B with an NVIDIA-A100-80GB GPU.

"Appendix H. More Visualizations" was really interesting to me. Thanks for the paper link.

---

👤 **louiehelm** commented the **2025-04-22** at **22:37:09**:<br>

It's fascinating how well your quants track optimal limits from rate-distortion theory.

Optimal R(D) = 2^(-2*bitrate)

![ik_graph_with_optimal2](https://github.com/user-attachments/assets/fac395df-f864-41b8-a131-044c44dc1022)

Some of your new quants actually dip down to only ~1.25 bits of overhead.

That's really good considering "optimal" = infinite codebook (which prob hurt t/s)

---

👤 **ikawrakow** commented the **2025-04-23** at **07:01:57**:<br>

Where does the equation for the optimal R(D) come from?

LLaMA-3 requires about ~1 bpw more to achieve the same quantization error compared to other models (see https://github.com/ikawrakow/ik_llama.cpp/discussions/8). Does this mean that the coding overhead there is < 0.5 bpw? Or does it rather mean that the model weights in LLaMA-3  do contain more information (which is my interpretation)?

---

👤 **saood06** commented the **2025-04-24** at **00:23:38**:<br>

>essentially what LLMs might become in the limit once they're trained hard enough to reach 100% entropy levels (a full 8.0 bits per byte)

Only some recent models are trained at FP8 (such as Deepseek V3/R1), they tend to be BF16, with FP4 training currently in the research stages see [this](https://arxiv.org/pdf/2501.17116)

---

👤 **saood06** commented the **2025-04-24** at **07:15:28**:<br>

Exllama-V3 added cache quantization, 

https://github.com/turboderp-org/exllamav3/commit/cf848114852240a51fb6b9e77c686051c39302b2

They also explain their reasoning in an issue copied below:

>So cache quantization is implemented now. It's a variant of the same technique used in V2, but now with separate bitrates (2-8 bpw plus 0.5 bpw of overhead) for K and V channels. Works a little better than in V2, and it's more flexible.
>
>I experimented with realtime trellis quantization, learned channel scales, autoencoders and more, but so far with little success, and not enough benefit to justify the overhead and complexity. There's still much to explore, though. For instance, I think it should be possible to learn an optimal rotation for the keys in a given layer, under a quantization constraint, then bake the same transformation into the Q and K projections, preserving their dot product.
>
>But for the time being, it's too much of a side quest, and I need to focus on some other stuff first. In the meantime you can get very usable results from k4v3 quantization, and more-or-less lossless quantization with k5v4. And it's "usable" down to k3v2, depending on the use case. Might make the model more creative or something, who knows (:. I still have to rig up some tests to see if it holds up over long contexts.

---

👤 **ikawrakow** commented the **2025-04-24** at **07:29:50**:<br>

> Does your new Trellis quant also have a +1.1bit gap between L2 70b and L3 70b?

I have not tried it for 70B models. It is too slow for the amount of patience I have. I know some people are OK spending 2 days quantizing a model on a GPU, but I'm not one of those.

---

👤 **ikawrakow** commented the **2025-04-24** at **08:18:08**:<br>

> Worst-case model weights can be approximated as maximally unpredictable Gaussian data -- essentially what LLMs might become in the limit once they're trained hard enough to reach 100% entropy levels

I'm not sure I can follow. On my book, LLMs only work because there are patterns encoded in the model weights, i.e., the model weights of an LLM are pretty much the opposite of a memoryless signal as required for these equations to hold. We also know that the model weights are definitely not Gaussian, and the so called "outliers" (i.e., weights that do not fall within the expectation of a normal distribution) are more important than the others. Also, the rate distortion equation tells us something about the difference between the signal (model weights) and its approximate representation (quantized model weights), but it tells us nothing about how this will affect observations (predicted token probabilities), which are the result of a complex set of linear and non-linear operations on the signal.