## 🗣️ [Discussion #319](https://github.com/ikawrakow/ik_llama.cpp/discussions/319) - KTransformers copying ik_llama.cpp

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-08 |
| **Updated** | 2025-04-13 |

---

## 📄 Description

[This PR](https://github.com/kvcache-ai/ktransformers/pull/754) is a direct copy from [this file](https://github.com/ikawrakow/ik_llama.cpp/blob/main/ggml/src/iqk/iqk_mul_mat.cpp) in `ik_llama.cpp`. It never acknowledges the source of the changes, and the KTransformers maintainers did not respond to [my comment](https://github.com/kvcache-ai/ktransformers/pull/754#issuecomment-2781515478) I left in the PR.

The PR is being sold as `IQ1_S` implementation, but it copies not just the `IQ1_S` GEMM, but also ~1800 LOCs of additional stuff, including the `IQ2_XXS` implementation, the new implementation of any float type x any other float type GEMM, and a bunch of other optimizations I have done since my contributions to [llamafile](https://github.com/Mozilla-Ocho/llamafile) ([394](https://github.com/Mozilla-Ocho/llamafile/pull/394), [405](https://github.com/Mozilla-Ocho/llamafile/pull/405), [428](https://github.com/Mozilla-Ocho/llamafile/pull/428), [435](https://github.com/Mozilla-Ocho/llamafile/pull/435), [453](https://github.com/Mozilla-Ocho/llamafile/pull/453), and [464](https://github.com/Mozilla-Ocho/llamafile/pull/464))

For those who don't know, KTRansformers uses the quantized GEMM/GEMV implementation that I contributed to [llamafile](https://github.com/Mozilla-Ocho/llamafile). `llamafile` uses the Apache-2.0 license, so I contributed the code under that license. KTransformers have kept the [copyright notice](https://github.com/kvcache-ai/ktransformers/blob/f4ae7c85edd66d6acf3ef253eeaf0143eb3358ab/third_party/llamafile/iqk_mul_mat.inc#L3) in  the file, but did not update after merging PR 754, which contains a copy of MIT licensed code.

KTransformers PR 754 is interesting anyway. Github user @godrosev entered issue [#209](https://github.com/ikawrakow/ik_llama.cpp/issues/209) on February 19 asking for `IQ1_S` support in `llamafile`. There was already implementation for the row-interleaved variant `IQ1_S_R4` in `ik_llama.cpp`, so I wasn't planning to also have support for `IQ1_S`, and suggested to them to use that instead. But after some back-and-fort, I decided to add `IQ1_S`, which I did in PR [#212](https://github.com/ikawrakow/ik_llama.cpp/issues/212) on Feb 20. The KTransformers PR 754 is on March 3 and comes from Github user @moonshadow-25. There are 5 commits in the PR, and the first 2 come from @godrosev. @godrosev and @moonshadow-25 both have no Github activity other the PR (and Issue [#209](https://github.com/ikawrakow/ik_llama.cpp/issues/209)). 

So now the question is, what do I do about that. Opinions?

---

## 💬 Discussion

👤 **moonshadow-25** commented on **2025-04-08** at **08:50:43**

hi ikawrakow, I am not an official developer of KT,@godrosv he is my colleague, and I am very sorry about this matter. After he gave me the code, I started the porting work without asking the source, but I noticed that the author in the file is also the same module's author as Llamafile, which is you. Afterwards, I completed all the porting work but did not modify any author information, because from the beginning KT kept mentioning that they used llamaflile as the core optimization, and I only filled in the complete functionality.

I have always felt that the CPU optimization in Llamafile is the best part done. If I really want others to not know that you did it, I can completely modify the variable or function names. However, I have fully ported it, only modifying the necessary interface parts, because I still believe that the iqk part of Llamafile is your contribution!

---

👤 **ikawrakow** commented on **2025-04-08** at **09:29:53**

> and I am very sorry about this matter

Are you planning to correct it? The 1800 lines added in your PR are not a "port", but a direct copy of portions of the code here. It would be very nice if the actual origin was acknowledged by you and by the KT developers.

---

👤 **moonshadow-25** commented on **2025-04-08** at **10:06:25**

Yes, I have always believed that both the early content and the “ported” parts of Llamafile originated from your work. And what I did more was porting and testing, so I never intended to modify (except for necessary interface adjustments) your work. I think this is your contribution！
I hope we can have more communication in the future

> 👤 **ikawrakow** replied on **2025-04-08** at **11:19:06**
> 
> Sorry, @moonshadow-25, but there are no "ported” parts of Llamafile in your PR. There are 1800 lines of code copied from here. They do not exist in Llamafile to be "ported" (i.e., copied) from there.
> 
> You have created a bit of a mess with your PR. KTransformers and Llamafile are both Apache-2.0 licensed. But the code here is published under a MIT License. Now, Apache-2.0 and MIT are both very permissive licenses, so it is easy to bundle code published under these license together, as explained for instance [here](https://infra.apache.org/licensing-howto.html). You could have even asked me if I would be willing to relicense the portions you copied to Apache-2.0 so it makes things easier for KTransformers (after all, I did change the MIT License of the code I contributed to Llamafile to Apache-2.0 to make it easier for them). But as permissive as these licenses are, it does not mean you can just ignore what they ask you to do.

> 👤 **moonshadow-25** replied on **2025-04-08** at **11:41:27**
> 
> Indeed, I am very sorry that I only realized the difference now. They look too similar, and both authors are you. So I subjectively assumed it was the same license.
> I must make some remedies as soon as possible, and I hope to hear your advice

---

👤 **ikawrakow** commented on **2025-04-13** at **15:56:21**

The KTransformers devs have now merged [this PR](https://github.com/kvcache-ai/ktransformers/pull/1116), which addresses the concern raised in this discussion => closing.