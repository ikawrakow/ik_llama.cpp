### 🗣️ [#166](https://github.com/ikawrakow/ik_llama.cpp/discussions/166) - Learning more LLM quantization

| **Author** | `robinnarsinghranabhat` |
| :--- | :--- |
| **Created** | 2025-01-05 |
| **Updated** | 2025-03-13 |

---

#### Description

For beginners like me to ML, I wanted to learn what research papers guided the quantization implement in llama.

It might sound silly but we have separate tricks for quantization during training and during evaluation right ?

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2025-01-05** at **10:37:28**:<br>

> For beginners like me to ML, I wanted to learn what research papers guided the quantization implement in llama.

I developed all quantization types in `llama.cpp` apart from the legacy quants `Q4_0, Q4_1, Q5_0, Q5_1, Q8_0` (but these are very simple round-to-nearest block-wise quants). I did not read any research papers, just went ahead and experimented. Rarely reading papers has always been my approach to research. I have found that reading what others have done influences my thinking direction and hence may prevent finding a better approach. I only go and read papers if I was not able to find a meaningful solution to a problem on my own.

> It might sound silly but we have separate tricks for quantization during training and during evaluation right ?  

`llama.cpp` does not do any training, so it is always post-training quantization (PTQ). But in general there is quantization-aware training (QAT), where the model is not actually quantized during training but model weights are forced to stay within a specified range with the hope that this will give better PTQ results. The only actually quantized model training approach I'm aware of is Bitnet from Microsoft Research, where a ternary model is trained (weights are -1, 0, 1, plus a per tensor float scaling factor). More recently researchers have been utilizing fine-tuning for PTQ, where some corpus of training data is used to guide PTQ (look for, e.g., Quip#, AQLM, QTIP). This is quite different from the simple quantization approaches used in `llama.cpp` and also here in this repository, requires a full-fledged training framework such as PyTorch, powerful GPU(s), and many hours/days of GPU time.

---

👤 **robinnarsinghranabhat** replied the **2025-01-10** at **21:38:11**:<br>

Thank you for this humble response ! 

Now I understand it's doing inference on quantized weights. 

But I get lost trying to understand llama cpp codebase. how should I navigate this codebase ?
I am comfortable with python, machine learning concepts and understand pointers in C.
 But never written complex programs in C/C++.

Do I need to understand fundamentals concept on operating systems, comp.arch, memory-management e.t.c. ? 

I want to be a programmar like you. 

Sorry .. lots of questions all over the place :(

> 👤 **arnfaldur** replied the **2025-03-13** at **02:10:31**:<br>
> Trying to understand this codebase isn't attacking the wall where it's lowest. You're probably best off finding some beginner/intermediate C++ courses online. I imagine that there are plenty available for free. You don't strictly need to understand all these fundamentals to understand what this project is doing, but you sound like you're in the *don't know what you don't know* phase and a general Computer Science course would likely get you the farthest at this point.