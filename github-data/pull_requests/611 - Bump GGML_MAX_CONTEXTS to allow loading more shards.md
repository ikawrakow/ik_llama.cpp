### 🔀 [#611](https://github.com/ikawrakow/ik_llama.cpp/pull/611) - Bump GGML_MAX_CONTEXTS to allow loading more shards

| **Author** | `Thireus` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-15 |
| **Updated** | 2025-07-16 |

---

#### Description

This var prevents more than 64 shards from being loaded - Specifically relevant for large models such as DeepSeek R1.

I have tested it extensively for a few weeks - see https://github.com/Thireus/ik_llama.cpp/commit/a66490410a366a9605234b94d67f3d9b7b389140

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High

---

#### 💬 Conversation

👤 **saood06** commented the **2025-07-15** at **01:19:45**:<br>

Would it make sense to also include this https://github.com/Thireus/ik_llama.cpp/commit/65dd65c10d2dc24cdddbd6255c3841c6a6c1038c as well for Windows users?

---

👤 **ikawrakow** submitted a review the **2025-07-15** at **05:08:20**: 💬 `COMMENTED`

---

👤 **saood06** submitted a review the **2025-07-15** at **05:12:32**: 💬 `COMMENTED`

---

👤 **saood06** commented during a code review the **2025-07-15** at **05:12:32** on `ggml/include/ggml.h`:<br>

It is if you want to use his tool suite, which makes use of GGUF split to this degree: https://huggingface.co/Thireus/DeepSeek-R1-0528-THIREUS-BF16-SPECIAL_SPLIT/blob/main/DeepSeek-TNG-R1T2-Chimera-THIREUS-BF16-00001-of-01148.gguf

1148 files for R1, so 2048 feels justified.

---

👤 **ikawrakow** submitted a review the **2025-07-15** at **05:59:36**: 💬 `COMMENTED`

---

👤 **ikawrakow** commented during a code review the **2025-07-15** at **05:59:36** on `ggml/include/ggml.h`:<br>

But apart from the tool suite, when are we going to need more than 64, or perhaps 256, shards?

Sure, the `ggml_context` struct is not that large (88 bytes, so we will waste a mere 170 kB).

But then again, are you actually having 1148 contexts **at the same time** in your tool suite?

---

👤 **Thireus** submitted a review the **2025-07-15** at **06:08:05**: 💬 `COMMENTED`

---

👤 **ikawrakow** commented the **2025-07-15** at **06:26:41**:<br>

How about this:
```c++
#ifndef GGML_MAX_CONTEXTS
#define GGML_MAX_CONTEXTS 64
#endif
```
along with a `cmake` variable that can be used to set `GGML_MAX_CONTEXTS`? You can then build the tool suite with whatever number of contexts you like (the way things are going, soon even 2048 may not be enough).  

I see that `GGML_MAX_CONTEXTS` is not used anywhere else apart from `ggml.c`, so strictly speaking it should not be the the `ggml` public API header (but this is of course not your fault or the issue handled by the PR).

---

👤 **saood06** submitted a review the **2025-07-15** at **06:26:49**: 💬 `COMMENTED`

---

👤 **saood06** commented during a code review the **2025-07-15** at **06:26:49** on `ggml/include/ggml.h`:<br>

>Of course if someone really wants to have less shards after downloading the mixture of shards, they can merge them, but that defeats the purpose of allowing for quick swaps between mixes by only downloading and replacing the necessary tensors.

I was just typing up a less eloquent version of this.

I like your tool, I am looking to adapt some of the recipes you found to kimi-k2 to fit on my 384 GB server.

---

👤 **Thireus** commented the **2025-07-15** at **06:35:54**:<br>

> How about this:
> 
> ```c++
> 
> #ifndef GGML_MAX_CONTEXTS
> 
> #define GGML_MAX_CONTEXTS 64
> 
> #endif
> 
> ```
> 
> along with a `cmake` variable that can be used to set `GGML_MAX_CONTEXTS`? You can then build the tool suite with whatever number of contexts you like (the way things are going, soon even 2048 may not be enough).  
> 
> 
> 
> I see that `GGML_MAX_CONTEXTS` is not used anywhere else apart from `ggml.c`, so strictly speaking it should not be the the `ggml` public API header (but this is of course not your fault or the issue handled by the PR). 

Still adds friction if users don't know they have to change it, so will need to be made explicit but I'm ok with this compromise since there aren't official pre-compiled versions here yet (less chance of people not knowing how to compile, and the Win binaries I distribute already come with 2048 set).

Thank you.

---

👤 **saood06** commented the **2025-07-15** at **06:58:21**:<br>

@ikawrakow 

>Which windows commit 

[Thireus@65dd65c](https://github.com/Thireus/ik_llama.cpp/commit/65dd65c10d2dc24cdddbd6255c3841c6a6c1038c)

>and when is there dynamic GGML_MAX_CONTEXTS?

And dynamic in the sense that built below 512 then nothing needs to be set, if built above 8192, set to only 8192 (as 8192 is the Windows limitation and 512 the default).

---

👤 **saood06** commented the **2025-07-16** at **00:31:03**:<br>

> [Thireus@65dd65c](https://github.com/Thireus/ik_llama.cpp/commit/65dd65c10d2dc24cdddbd6255c3841c6a6c1038c) would be a separate pull request as this is a different limitation (OS limitation for number of opened files), that code is required for Windows while other platforms (linux, macos) can use ulimit to lift the limitation.

Sounds good to me.

---

👤 **ikawrakow** submitted a review the **2025-07-16** at **12:11:08**: ✅ `APPROVED`

---

👤 **ubergarm** commented during a code review the **2025-07-16** at **13:36:21** on `ggml/include/ggml.h`:<br>

@saood06 I have now [uploaded a few Kimi-K2s](https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF). A couple of which might suit your needs.

So if I understand @Thireus approach better it is to essentially pull apart individual tensors quantized to different levels and mix-match them back together using a bunch of "shards"?

If so that kinda makes sense, given most of my quants are using the same attn/shexp/ffn dense layers and only changing the exps more or less.

Feel free to rip tensors out of my GGUFs and frankenstein back together another mix! Interesting...

---

👤 **ubergarm** submitted a review the **2025-07-16** at **13:36:22**: 💬 `COMMENTED`

---

👤 **Thireus** submitted a review the **2025-07-16** at **14:35:16**: 💬 `COMMENTED`

---

👤 **ubergarm** submitted a review the **2025-07-16** at **14:42:59**: 💬 `COMMENTED`

---

👤 **ubergarm** commented during a code review the **2025-07-16** at **14:42:59** on `ggml/include/ggml.h`:<br>

frankenshards i love it! it is still a bit beyond my full conception with the working parts. it'd be cool to see a 5 minute demo video if such a thing is possible. i'll have to look closer when I get some more time. thanks for thinking so far out there and pushing the innovation!

---

👤 **ikawrakow** submitted a review the **2025-07-16** at **14:52:59**: 💬 `COMMENTED`

---

👤 **Thireus** submitted a review the **2025-07-16** at **15:30:40**: 💬 `COMMENTED`

---

👤 **Thireus** commented during a code review the **2025-07-16** at **15:30:40** on `ggml/include/ggml.h`:<br>

> What if one wants a different imatrix? Or if there is an improvement in the quantization function?

They'll create their own shards with [DeepSeek-R1-0528-THIREUS-ANY-SPECIAL.sh](https://github.com/Thireus/GGUF-Tool-Suite/blob/main/models/DeepSeek-R1-0528/DeepSeek-R1-0528-THIREUS-ANY-SPECIAL.sh). And adjust [download.conf](https://github.com/Thireus/GGUF-Tool-Suite/blob/main/models/DeepSeek-R1-0528/download.conf) to point to their repos.