### 🗣️ [#82](https://github.com/ikawrakow/ik_llama.cpp/discussions/82) - 4bpw GGML TYPE?

| **Author** | `Nexesenex` |
| :--- | :--- |
| **Created** | 2024-10-07 |
| **Updated** | 2024-10-17 |

---

#### Description

Hey IK,

It's been a while you forked, and I wondered if you'd be willing to PR something close to a 4 bpw (3.8125-4.0625?, I don't know) ggml type on LlamaCPP, if you have one viable in store. The gap between IQ3_S and IQ4_XS is huge, and there are some reported problems with IQ3_S and IQ3_XXS), which can screw hybrid IQ4_XS based quants where attn_q and attn_output (or some layers of ffn gate and up) are passed in IQ3_S to fit in some VRAM configs.

Maybe with Johannes Gaessler's goodwill, It would make full offload of the 123b parameters viable on 64GB VRAM, and the 70b models viable on 36GB VRAM.

More broadly, your work is sorely missing on LCPP.

Cheers!

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2024-10-08** at **05:17:48**:<br>

Hey Nexes the Old, did you try `IQ3_K` and `IQ4_K`? I think a mix of these two will give you what you want, and it will be better than what you could do with i-quants in `llama.cpp`.

---

👤 **ikawrakow** replied the **2024-10-08** at **10:55:49**:<br>

@Nexesenex 

Here is an example - LLaMA-3.1-8B-Instruct. We look at `PPL(Q)/PPL(fp16)-1` for a context of 2048 (but note that the `PPL(Q)/PPL(fp16)` ratio is almost independent of context length). First a graph with all quants, including the new `IQX_K` quants in cyan,  using a logarithmic y-axis to get the big picture. The two magenta circles that sit around 4 bpw are mixes between `IQ3_K/IQ4_K/IQ5_K/IQ4_XS`. To me it looks like they are pretty much on the quantization error vs model size Pareto front that we can get from i-, k-, and iqk-quants (and i- and iqk-quants are pretty much as good as it gets without additional fine tuning).
  
![il31_8B](https://github.com/user-attachments/assets/4127966d-0d3d-4ee3-926c-c9eaa18461f1)

Then a zoomed-in graph in the bpw area of interest with a linear y-axis.
![il31_8B_nesenex](https://github.com/user-attachments/assets/cbdd834b-bd66-47e0-aa9e-6e17f82286d4)

The two magenta mixes are at 4.0 and 4.09 bpw. These are bpw that include token embedding and output tensor. The token embedding tensor is quantized with `IQ3_K`, the output tensor `output.weight` with `Q6_K`. In the case of LLaMA-3.1 with its 128k vocabulary `output.weight` is quite large, and hence increases the effective bpw by 0.167 bpw (compared to it being ignored, as quantization literature tends to do, or it being quantized with 4 bpw). Hence, for a larger model where the output tensor represents a much smaller fraction of the overall model size, these mixes will be sub-4 bpw. The smaller mix is composed as follows
* `output` - `Q6_K`
* `token_embd, attn_q, attn_k, ffn_gate` - `IQ3_K`
* `attn_v` - `IQ5_K`
* `attn_output` - `IQ4_K`
* `ffn_down, ffn_up` - half with `IQ3_K`, other half with `IQ4_K` (using function `use_more_bits(i_layer, n_layer)` to select `IQ4_K` vs `IQ3_K`

The larger mix is as the above, but in addition uses
* `ffn_gate` - half with `IQ4_XS`, other half `IQ3_K`, again using `use_more_bits(i_layer, n_layer)`

I can add one of these. Let me know if you prefer the smaller or the larger one.

---

👤 **ikawrakow** replied the **2024-10-09** at **09:54:18**:<br>

See #83

---

👤 **Nexesenex** replied the **2024-10-09** at **14:58:25**:<br>

Hey IK,

I was about to answer you, but of course, you made some magic happen already.

Fantastic work, as always. A new SOTA 4.25BPW GGML_TYPE quant is a huge boost. Can it be integrated in the official LlamaCPP by moving the relevant section of your ik files in the traditionnal equivalents in LCPP official?

As for quant mixes, on LCPP official, I passed attn_v in Q6_K and attn_K in Q5_K for my >IQ3_M and IQ4_XS mixes when vocab is above 128000. The ppl usually drops by more than 0.01, I suspect it might help other indicators even more, for 180MB on Llama 3 70b and ulterior, that's a good trade.

I also generally beef up to the higher quant the first and last layers attn_k, attn_q, and ffns in all cases, because they are either the closest from embeddings (as you were doing already on several quant mixes), or the last ones before the final output. 

I use an equivalent IQ3_XXL mix to your IQ3_KL. on the top of a bumped ffn_down, I'll bump ffn_up more than ffn_gate to see if it brings a bonus compared to equalizing them, I used several variants of your more_bits function to achieve steps of 12.5% layers quantized to the higher quant accordingly to my needs.

What I was wondering about is a LCCP official mergeable IQ4_XXS / IQ4_K_"XXS" GGML type (tensor level quant), at 4-4.0625bpw, if such thing is possible and viable compared to a IQ3/IQ4 mix, to get rid of the IQ3_S I'm using, because on some models they are worst than Q3_K (Miqu attn_q and attn_output, for example, I observed some discrepancy on Qwen2 72b as well).

I speak about LCPP official, because I was.. unable to compile IK_Llama on MSVS, and I need official as the base for my fork of KoboldCPP, the inference software I modified and use with everything, rebasing it on your IK LLama while I can't even compile it seems unviable to me. Moreover, I do not know your personal objectives nor relations with the LCPP official project, but a broad compatibility for your quants would allow people to.. use them, and not waste compute, energy, and time on non-SOTA quants for their models.

---

👤 **ikawrakow** replied the **2024-10-09** at **16:23:12**:<br>

> Can it be integrated in the official LlamaCPP...

The license is MIT, so obviously it can be integrated into mainline `llama.cpp`. Will I do it? Of course not.

> I speak about LCPP official, because I was.. unable to compile IK_Llama on MSVS, and I need official as the base for my fork of KoboldCPP, the inference software I modified and use with everything, rebasing it on your IK LLama while I can't even compile it seems unviable to me. 

You could have opened an issue, no? With the output of the build process. I don't have access to a Windows box and Windows is certainly not my priority, but sometimes one can fix it just from the compiler error messages.

> Moreover, I do not know your personal objectives nor relations with the LCPP official project, but a broad compatibility for your quants would allow people to.. use them, and not waste compute, energy, and time on non-SOTA quants for their models.

My personal objective is to have fun :smiley:
 
Quants are kind of orphaned in mainline and have become a "commodity", with tons of low quality quantized models being distributed on HuggingFace as GGUFs. Hence, people interested in (high quality) quantization work are better off here than mainline. Or people running on the CPU. Or people using models that run much faster here than in mainline also on the GPU (e.g., Gemma), etc. I do sync with mainline from time to time, but I did not see anything worth merging since I last synced in August. Am I missing something from mainline that you find essential? 

> I use an equivalent IQ3_XXL mix to your IQ3_KL. on the top of a bumped ffn_down, I'll bump ffn_up more than ffn_gate to see if it brings a bonus compared to equalizing them, I used several variants of your more_bits function to achieve steps of 12.5% layers quantized to the higher quant accordingly to my needs.

Sure, one can spend a lot of time experimenting. I see your PR 8917 in mainline has not been merged. As I believe that having a more flexible and convenient way to specify quantization mixes is definitely worth having, your PR is likely to be more successful here than there.

---

👤 **Nexesenex** replied the **2024-10-17** at **04:04:29**:<br>

I submitted my PR 8917 here, as invited to.

As for mainline, there's nothing essential for me since august, aside for maintaining some sort of compatibility with KCPP so I can attempt a rebase on your fork without breaking my head too hard, even if that might still be too hard. :D

A PR maybe worth testing is this one, with several percents boost in PP & TG on my side on Cuda :  https://github.com/ggerganov/llama.cpp/pull/8366

For the compile problem, I could have opened an issue but I was a bit discouraged by the idea that I could not even use your quants for my use (KoboldCPP + ST, I look at Lollms with curiosity also). My bad, but a white knight came to fix that a day before a lovely IQ4_KSS appeared, so here I am, llama-server + ST it is for now.

As for the beef with mainline, well, I really regret that the quality and speed of inference went maybe a bit low into the priority list. It seemed already to be the case when Johannes Gaessler developed the first KV quant 8 bits in late 2003. Anyway, I'm glad you keep having fun by blowing up the charts. Your work is really phenomenal, and I wish that your quants became the new baseline of the GGUF side of Hugging Face.

But where would be the fun in that? :X