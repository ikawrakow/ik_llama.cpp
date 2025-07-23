### 🗣️ [#543](https://github.com/ikawrakow/ik_llama.cpp/discussions/543) - dots.llm1 support and thanks

| **Author** | `Iconology` |
| :--- | :--- |
| **Created** | 2025-06-20 |
| **Updated** | 2025-07-03 |

---

#### Description

Hey, friend,

Out of curiosity, do you have any plans to add dots.llm1 support? The model seems interesting enough. I tried it out on mainline, but the speeds were atrocious for its size, making it unusable, at least for me. That’s why I jumped over to your fork (thanks to ubergarm) for both the insane MoE speedups and for being the godfather of, arguably, the absolute SOTA quants in my eyes.

Here's the pull request from mainline for dots:
https://github.com/ggml-org/llama.cpp/commit/9ae4143bc6ecb4c2f0f0301578f619f6c201b857

---
Regardless of whether it’s on your roadmap or not, I just wanted to say thank you, ikawrakow, for all that you have done and continue to do. You are one of a kind.

---

#### 🗣️ Discussion

👤 **saood06** replied the **2025-06-20** at **03:21:14**:<br>

>The model seems interesting enough.

I agree, from a quick skim of the PR code, I don't see anything that would lead to a complicated port. I could do it if no one else gets to it first.

Especially due to this part in that PR:

>The model architecture is a combination of Qwen and Deepseek parts, as
seen here:
>
>https://github.com/huggingface/transformers/blob/ffe12627b4e84489d2ab91dd0ec00614855edc79/src/transformers/models/dots1/modular_dots1.py

> 👤 **firecoperana** replied the **2025-07-02** at **22:56:45**:<br>
> @saood06 Are you working on it? If not, I can give a try.
> 
> 👤 **saood06** replied the **2025-07-03** at **02:23:35**:<br>
> #573 exists now. Testing is welcome.