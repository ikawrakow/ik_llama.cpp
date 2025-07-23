### 🔀 [#449](https://github.com/ikawrakow/ik_llama.cpp/pull/449) - Legacy quants conversion schemes in convert_hf_to_gguf.py

| **Author** | `Nexesenex` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-23 |
| **Updated** | 2025-05-24 |

---

#### Description

This, notably in order to make smaller conversions to generate an iMatrix file.

`Q4_0`,`Q4_1` are here using embeddings, output, attn_k and attn_v in q5_0.
`Q5_0`,`Q5_1` are here using embeddings, output, attn_k and attn_v in q8_0.

Adapted from the following llama.cpp mainline PR : https://github.com/ggml-org/llama.cpp/pull/9022 Original author @chentyjpm

Reason : Even in pure q4_0, an iMatrix is viable (much less than 0.01 ppl difference with a q8_0 one on the final quantization made with the created iMatrix).
Those schemes are thus pertinent imho.
I personally use the q5_0 scheme to make my iMatrixes for the L3 70b models, and the ppl difference is less than 0.005 on the final quantized model with iMatrix, this compared to an f16 iMatrix made by Bartowski or Mradermacher.

Also, 2 forgotten mentions of FTYPE IQ3_KL are added in llama.cpp file, and one IQ5_KS mention in the mmvq_type_supported switch.

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High

---

#### 💬 Conversation

👤 **Nexesenex** commented the **2025-05-23** at **14:38:10**:<br>

Well, when I test a new finetune or merge of a big model I can't run in 16 or even 8 bits, I like to make a simple q5_0 or even q4_0 conversion to test it in chat in full offload or quasi-full offload on my 64GB VRAM.

- If the model doesn't please me, I didn't bother to make a fp16 gguf, then a quantized gguf, and I simply ditch the smaller conversion and the HF weights. That's the convenient part, it was fast, easy, and disk-space-savvy

- If it pleases me, I use that same small conversion to make the iMatrix in full offload or quasi-full offload, then ditch the conversion, make my fp16 (or even 8 bits) gguf, ditch the HF weights, and make the final quant I want out of the q8_0/fp16 gguf and the iMatrix previously made out of the small conversion. That's the incidental part.

I think some other folks could use that too, especially the ability to convert and test a finetune or merge of a supported foundation model in a single shot without bothering with the usual 2 steps approach (source gguf, then quant).

---

👤 **ikawrakow** commented the **2025-05-23** at **15:23:12**:<br>

Did you test that the conversion is working? I'm in the middle of something and don't feel like downloading a few models from HF to test.

The described new model testing procedure saves 1 conversion to `bf16` (or `Q8_0`) for the discarded models that you didn't like. Have you considered the possibility that you are discarding good models because the `Q4_0` conversion without an imatrix has gone sideways (and this is the actual reason you are not liking the model, not the model itself)?

---

👤 **Nexesenex** commented the **2025-05-23** at **16:42:03**:<br>

> Did you test that the conversion is working? I'm in the middle of something and don't feel like downloading a few models from HF to test.

The Llama 3 class models are working, that's certain.

Yes, the conversion is working in q4_0, q4_1, q5_0, q5_1. I use q4_0 and q5_0 very often, I'm just sharing the code I edited made months ago.
The tensor rules are working also. Embed, Output, K and V are converted in the upper quant as instructed so the quantizations are viable.

If you were inclined to implement q6_0 in the .py conversion during a day of schedule haze and benevolence, that would be even better, of course, because q8_0 is a bit overkill for such tests or iMatrix creation! ^^

> The described new model testing procedure saves 1 conversion to `bf16` (or `Q8_0`) for the discarded models that you didn't like. Have you considered the possibility that you are discarding good models because the `Q4_0` conversion without an imatrix has gone sideways (and this is the actual reason you are not liking the model, not the model itself)?

I made perplexity tests on my conversions (more than a hundred of them during the last trimester), and they are as expected. For example for a L3 70b with a perplexity of 3.83 in FP16, I will have around 3.91 on a converted q5_0 mix such as proposed. The quality is fine on the fine models. A q4_0 with Embeddings, Output, K and V in q5_0 is still acceptable without an iMatrix as well for testing and iMatrix purpose.

Unless I'm thrilled with a model, I keep using some of those conversions as they are when I have enough VRAM, not even bothering to make the whole imat/f16/quant process.

When I'll come home tonight, I'll make some tests beyond the Llama 3 70b I've been converting extensively with that method.

---

👤 **ikawrakow** submitted a review the **2025-05-24** at **06:09:15**: ✅ `APPROVED`