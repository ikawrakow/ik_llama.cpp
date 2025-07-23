### 🔀 [#114](https://github.com/ikawrakow/ik_llama.cpp/pull/114) - MMQ Kernel for Q6_0 (pretty please!)

| **Author** | `Nexesenex` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-11-20 |
| **Updated** | 2024-11-20 |

---

#### Description

Q6_0 MMQ Kernel attempt.

Of course, if I can reproduce the formatting, compile and run it, I don't understand anything to the maths involved within the main template, and thus, perplexity jumps by a factor 30000 on a pure Q6_0 quant. :D

I used q5_0 as a base.

I know you're not very much into making MMQ Cuda Kernels, but could you please do this one if it's not too bothersome, IK? Qwen2 models are quite popular and good, but their ffn_down tensors have a reversed shape, and thus, need either Q5_1 as a fallback, either Q8_0, which is unsatisfactory in both case for the ratio quality/size of an overall 5-6 bpw quant.

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [ ] Low
  - [ ] Medium
  - [x] High (it runs, but perplexity is 200k with force MMQ on a pure Q6_0 Sheared Llama 2 2.7b), instead of the 7-8 expected, and it's way above my league to fix that.

---

#### 💬 Conversation

👤 **ikawrakow** submitted a review the **2024-11-20** at **09:24:50**: 💬 `COMMENTED`

---

👤 **Nexesenex** submitted a review the **2024-11-20** at **15:21:54**: 💬 `COMMENTED`

---

👤 **Nexesenex** commented during a code review the **2024-11-20** at **15:21:54** on `ggml/src/ggml-cuda/mmq.cuh`:<br>

It's hard. Too hard for me still. :)

I don't find a similar template for Q5_0 Cublas in convert.cu, or anything remotely close, so I kept digging if I could find similar and sufficient patterns on another quant, or in common.cuh to have a delta and understand how to transpose. I didn't find what I needed. I am sorry. ^^