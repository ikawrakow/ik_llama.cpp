### 🔀 [#117](https://github.com/ikawrakow/ik_llama.cpp/pull/117) - Some minor quant strategies tweaks

| **Author** | `Nexesenex` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2024-11-22 |
| **Updated** | 2024-11-23 |

---

#### Description

Here's what I'd suggest for starters :

- Rationalize Q2_K_S ffn_down and attn_v (+1% size, -2.5% ppl)

- Bump attn_v and attn_k for Q2_K_S and Q2_K if GQA>=2. Uncripple attn_k for IQ3_XXS / IQ3_XS if GQA>=2
-> Gemma v2 (GQA2) is popular and sensitive to both. L3 models as well.

- Apply 8 experts rules to :
  - MOEs with more than 8 experts..
  - MOEs with 4 experts which should be treated as 8 considering that their shared tensors relative size is already low compared to their ffn tensors).
  - models with 2 or more experts (such Frankenstein hybrids are published on HF with 2 experts, let them have MOE quants equivalent in bpw to standard models).

- Rationalize MOEs attn_k and attn_v for the 1 & 2 bit IQ quants, and attn_q for 1,2 and small 3 bpw quants.

- Rationalize attn_ouput for IQ2_XXS, IQ2_XS, IQ2_S and IQ2_M (IQ3_XXS is sufficient), in respect for what was done for the IQ1 quants, themselves shrunk in IQ2_KS. (no tests made today except for IQ2_S and M, it's mere common sense).

- rationalize the ffn_down on IQ2_S and IQ2_M. (size is equivalent with the attn_output shrink, ppl drops by 0.5%).

Test made today on Sheared Llama 2.7b, but I use those recipes among others for a long time already;


Further ideas for a subsequent PR :

- IQ and IQ_K should maybe not be mixed together unless they are switchable 1:1 on all the supported hardware, accounting also for those having a Cuda MMQ kernel available and those which don't.

- Maybe also the IQ1 IQ2 tree should be dismantled and spread into the tensor trees like every other quants.


- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2024-11-22** at **15:30:05**:<br>

Can you provide some data to support these changes?

---

👤 **Nexesenex** commented the **2024-11-22** at **16:53:59**:<br>

Not really, IK, i'd have to remake all tests I did during the previous months. I never knew how to log properly LlamaCPP data, so I accumulated knowledge and edits along the way and just restitute you the simplest part of it. I submit that to you in a "trust me bro" fashion because I suppose that you know what I know and then some, and just have more interesting things to do with your skillset than to mess hamster-style with quant strategies like I did since early 2024.

Broadly, there's a few principles that I discovered through your work :

- Most of models will receive well the following structure around a GGML_type (with -2 (lower bpw quant) to +2 (higher bpw quant) degrees of quantization around the base ggml_type) :

- Attn_q : basetype -1 or -2
- Attn_k : basetype or +1 (you go on -1 sometimes, I tend to disagree with that)
- Attn_v : basetype +1 or +2. The higher the GQA, the more interesting the bump is, nothing new.
- attn_output : basetype +1 for 1-2bpw, basetype for 3bpw, basetype -1 for 4bpw or more. (ex : 3.5 bpw attn_output for 2.5bpw ftype doesn't show any benefit compared to a slight bump of ffn_down, for example).
- ffn_down : basetype +1 as much as possible, especially the first and last eighth of layers, model archs sensitivity are differing vastly for the intermediate layers. Going +1 or +1.5bpw for 1/8 of the layers, instead of +0.5bpw for 3/8 (2 first eights, one last eight or the opposite) of the layers is overkill, especially if the attention tensors are not calibrated for that on the affected layers. 
- ffn_gate and up are more tricky, but nevertheless the first / last layers bump applies too, especially since L3 models which are more "dense" than their predecessors.
- embedding and output, the bigger the base weight is, the more you can quantize it, nothing new. High vocab and monolithic embed/output answer to this.
MOES : 2 experts allow already a bump on the attn tensors, including q and output.
4 experts should really be treated like 8 experts models, there's no reason at all to discriminate them because they operate the very same (2 experts active), I noticed that on those Pivot/Solar 4 experts model.

So, without any disrespect, pick what you like, I'm sure that some of it makes sense to you, and ditch what's "too much" for your taste.

And if you'd like me to go on with the quant strategies, please tell me, I'd be glad to help on something that I actually can grasp and have experience upon.

Here's for you to eventually get a look on some experiments I made so you can check how far I went : 07ad6c6f321ea3643cff5d38766ce8f13a785bfcmaster_loot_2/