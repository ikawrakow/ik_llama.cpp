### 🔀 [#96](https://github.com/ikawrakow/ik_llama.cpp/pull/96) - Quant strategies: attn_q Q4 & attn_v Q6 for Llama 3.1 Q5_K_S

| **Author** | `Nexesenex` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-19 |
| **Updated** | 2024-11-22 |

---

#### Description

Pattern (attn-q -1 attn-v+1) worth to be tested on more quants levels (Q_x_K, IQx, & IQx_K) and on Llama 3.0 if confirmation is needed.

PPL 512 = -0.024 for 70b ; - 0.005 for 8b
Size = - 640MiB for 70b ; - 64MiB for 8b

70b Q5_K_S now beats Q5_K_M by -0.012 ppl, with the same source bf16 and imatrix.

I suspect that it goes similarly for L3 as well, which was quite insensitive to attn_q quantization as I discovered when I made my IQ3_L quant strategies for my own use.

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High

---

#### 💬 Conversation

👤 **ikawrakow** submitted a review the **2024-10-19** at **15:24:19**: ✅ `APPROVED`<br>

Yes, reducing bpw for `attn_q` and increasing `bpw` for `attn_v` is a good strategy to improve quantized model performance in general in my experience.

---

👤 **Nexesenex** commented the **2024-10-19** at **16:04:22**:<br>

If you're open to the idea, I can contribute more to that quant strategy part, in a progressive way, PR by PR.

I now handle well the afferent code, and got a lot of experimentation behind me already.

---

👤 **ikawrakow** commented the **2024-10-20** at **09:18:46**:<br>

> If you're open to the idea, I can contribute more to that quant strategy part, in a progressive way, PR by PR.
> I now handle well the afferent code, and got a lot of experimentation behind me already.
> The merged PRs/commits can then be squashed to keep the commit log clear of clutter.

Sure, go ahead.

---

👤 **Nexesenex** commented the **2024-10-20** at **22:44:46**:<br>

Shall I separate the IQ_K from the legacy IQ Quants in the mixes?

---

👤 **Nexesenex** commented the **2024-11-22** at **07:41:35**:<br>

@ikawrakow would it be possible and not a hassle for you to decouple the quant strategies part of the llama.cpp source file in order to reduce the recompilation time when the quant strategies are edited, so it can speed up the tests?

---

👤 **ikawrakow** commented the **2024-11-22** at **08:08:37**:<br>

It is of course possible. But is compilation time really a major factor in testing? One needs to quantize and run a test such as PPL. Compared to that `llama.cpp` compilation time should not be a major factor. Or am I missing something?

---

👤 **Nexesenex** commented the **2024-11-22** at **11:22:35**:<br>

Well, if one plays with use more bit formulas (I use customized ones a lot), which are not supported by the CLI args, then the endless lengthy recompiles quickly become a hassle. ^^

---

👤 **ikawrakow** commented the **2024-11-22** at **16:40:39**:<br>

So, let's say compiling `llama.cpp` takes 15 seconds. Quantizing a 7B model is 15+ seconds. Running PPL is 60 seconds. So, at the very best, compilation time is ~15% of the overall time to test. If we are looking at larger models and/or more than one model (my usual approach is to check at least 5 models before drawing conclusions that one quantization strategy is better than another), the compilation time basically becomes a negligible fraction of the time needed to test a new quantization strategy. 

I can see that compiling `llama.cpp` may become annoying if one is quantizing and running a single small model (e.g. LLaMA-3.2-1B). But that's not what one should be doing to change a quantization strategy.