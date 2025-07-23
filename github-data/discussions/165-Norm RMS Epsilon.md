### 🗣️ [#165](https://github.com/ikawrakow/ik_llama.cpp/discussions/165) - Norm RMS Epsilon

| **Author** | `Nexesenex` |
| :--- | :--- |
| **Created** | 2024-12-25 |
| **Updated** | 2024-12-27 |

---

#### Description

While it crosses my mind..

@Ikawrakow : a while ago, you made some measurements with variations of Norm RMS Epsilon which showed some little benefits to offset it for <2bpw quants. It was on L2 I believe, and I wonder if it applies to other arches, and if yes, if there's some sort of "formula" which would come with it to improve the low bitrate quants themselves.

Just beotian thoughts.

And merry XMAS btw, if you celebrate it!

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2024-12-27** at **17:44:24**:<br>

I'm travelling, so just quickly from the phone.

Yes, there is a small benefit from increasing rms_eps also for LlaMA-3, but only for very low-bit quants (IQ2_XXS). No, I haven't done any kind of systematic investigation.