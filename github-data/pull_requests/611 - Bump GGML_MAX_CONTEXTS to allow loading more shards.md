## 🔀 [Pull Request #611](https://github.com/ikawrakow/ik_llama.cpp/pull/611) - Bump GGML_MAX_CONTEXTS to allow loading more shards

| **Author** | `Thireus` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `patch-1` |
| **Target Branch** | `main` |
| **Created** | 2025-07-15 |
| **Updated** | 2025-07-16 |
| **Merged** | 2025-07-16 |

---

## 📄 Description

This var prevents more than 64 shards from being loaded - Specifically relevant for large models such as DeepSeek R1.

I have tested it extensively for a few weeks - see https://github.com/Thireus/ik_llama.cpp/commit/a66490410a366a9605234b94d67f3d9b7b389140

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High

---

## 💬 Conversation

👤 **saood06** commented on **2025-07-15** at **01:19:45**

Would it make sense to also include this https://github.com/Thireus/ik_llama.cpp/commit/65dd65c10d2dc24cdddbd6255c3841c6a6c1038c as well for Windows users?

---

👤 **ikawrakow** started a conversation on `ggml/include/ggml.h` on **2025-07-15** at **05:08:20**

Is 2048 really needed? The quoted whisper.cpp thread talks about 256 contexts, not 2048.

> 👤 **saood06** replied on **2025-07-15** at **05:12:32**
> 
> It is if you want to use his tool suite, which makes use of GGUF split to this degree: https://huggingface.co/Thireus/DeepSeek-R1-0528-THIREUS-BF16-SPECIAL_SPLIT/blob/main/DeepSeek-TNG-R1T2-Chimera-THIREUS-BF16-00001-of-01148.gguf
> 
> 1148 files for R1, so 2048 feels justified.

> 👤 **ikawrakow** replied on **2025-07-15** at **05:59:36**
> 
> But apart from the tool suite, when are we going to need more than 64, or perhaps 256, shards?
> 
> Sure, the `ggml_context` struct is not that large (88 bytes, so we will waste a mere 170 kB).
> 
> But then again, are you actually having 1148 contexts **at the same time** in your tool suite?

> 👤 **Thireus** replied on **2025-07-15** at **06:08:05**
> 
> Sharding with max 1 tensor per shard, which allows each tensor to be swapped individually at will. So one can create quants of individual tensors, and moving from one mixture of quants to another for a specific model simply means swapping some shards for others. No quantisation necessary, only download.
> 
> It works quite well and saves a tone of time, as one quand quickly swap tensor quants without going through quantising the whole model again. But it would be somewhat also effective if we could also quantise individual tensors (if someone wants to create such alternative tool, or enhance llama-quantization to allow this), which would give an alternative when shards aren't available.
> 
> https://huggingface.co/collections/Thireus/deepseek-r1-0528-thireus-special-split-68725429aceffbd1094bdd29
> 
> Of course if someone really wants to have less shards after downloading the mixture of shards, they can merge them, but that defeats the purpose of allowing for quick swaps between mixes by only downloading and replacing the necessary tensors.
> 
> I wrote a downloader that managed this all which is quant_downloader.sh at the root of gguf.thireus.com if you'd like to try it out. It can be used for any existing recipes, including @ubergarm ones (since he's one of the few who shares his recipes openly), providing all the quants of the tensors that comprise the recipe are available somewhere to download.
> The vision is that quantising models becomes more efficient, with one person pre-quantising tensors individual into shards and sharing recipes only, instead of sharing whole merged models (which can always be provided as an option for the users who really hate optimisation).
> 
> Didn't want to advertise my tool specifically, as I believe there are or can be other use cases and other tools that would benefit from an increased context size, as the upvotes from the llama.cpp seem to suggest.

> 👤 **saood06** replied on **2025-07-15** at **06:26:49**
> 
> >Of course if someone really wants to have less shards after downloading the mixture of shards, they can merge them, but that defeats the purpose of allowing for quick swaps between mixes by only downloading and replacing the necessary tensors.
> 
> I was just typing up a less eloquent version of this.
> 
> I like your tool, I am planning to adapt some of the recipes you found to kimi-k2 to fit on my 384 GB server.

> 👤 **ubergarm** replied on **2025-07-16** at **13:36:21**
> 
> @saood06 I have now [uploaded a few Kimi-K2s](https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF). A couple of which might suit your needs.
> 
> So if I understand @Thireus approach better it is to essentially pull apart individual tensors quantized to different levels and mix-match them back together using a bunch of "shards"?
> 
> If so that kinda makes sense, given most of my quants are using the same attn/shexp/ffn dense layers and only changing the exps more or less.
> 
> Feel free to rip tensors out of my GGUFs and frankenstein back together another mix! Interesting...

> 👤 **Thireus** replied on **2025-07-16** at **14:35:16**
> 
> @ubergarm Yes. Each tensor of the model is quantised to be exactly 1 shard, see [this collection](https://huggingface.co/collections/Thireus/deepseek-r1-0528-thireus-special-split-68725429aceffbd1094bdd29) and for example this [tensors.map](https://huggingface.co/Thireus/DeepSeek-R1-0528-THIREUS-BF16-SPECIAL_SPLIT/blob/main/tensors.map) for BF16. All possible combinations of recipe can then be produced (providing these shards are available).
> 
> Users can use [quant_recipe_pipeline.ipynb](https://colab.research.google.com/github/Thireus/GGUF-Tool-Suite/blob/main/) to compute the recipe suitable to their VRAM and RAM requirements. Or use existing recipes such as yours.
> 
> Once the recipe is generated, they can download the corresponding shards (tensors) using [quant_downloader.sh](https://github.com/Thireus/GGUF-Tool-Suite/blob/main/)
> 
> So, there is no need to quantize models anymore, only download the mixture of shards as defined in the recipe. Users also don't need to merge the shards, thanks to this pull request llama can load all the individual frankensteined 1148 shards.

> 👤 **ubergarm** replied on **2025-07-16** at **14:42:59**
> 
> frankenshards i love it! it is still a bit beyond my full conception with the working parts. it'd be cool to see a 5 minute demo video if such a thing is possible. i'll have to look closer when I get some more time. thanks for thinking so far out there and pushing the innovation!

> 👤 **ikawrakow** replied on **2025-07-16** at **14:52:59**
> 
> > So, there is no need to quantize models anymore, only download the mixture of shards as defined in the recipe. Users also don't need to merge the shards, thanks to this pull request llama can load all the individual frankensteined 1148 shards.
> 
> What if one wants a different imatrix? Or if there is an improvement in the quantization function?

> 👤 **Thireus** replied on **2025-07-16** at **15:30:40**
> 
> > What if one wants a different imatrix? Or if there is an improvement in the quantization function?
> 
> They'll create their own shards with [DeepSeek-R1-0528-THIREUS-ANY-SPECIAL.sh](https://github.com/Thireus/GGUF-Tool-Suite/blob/main/models/DeepSeek-R1-0528/DeepSeek-R1-0528-THIREUS-ANY-SPECIAL.sh). And adjust [download.conf](https://github.com/Thireus/GGUF-Tool-Suite/blob/main/models/DeepSeek-R1-0528/download.conf) to point to their repos.

---

👤 **ikawrakow** commented on **2025-07-15** at **06:26:41**

How about this:
```c++
#ifndef GGML_MAX_CONTEXTS
#define GGML_MAX_CONTEXTS 64
#endif
```
along with a `cmake` variable that can be used to set `GGML_MAX_CONTEXTS`? You can then build the tool suite with whatever number of contexts you like (the way things are going, soon even 2048 may not be enough).  

I see that `GGML_MAX_CONTEXTS` is not used anywhere else apart from `ggml.c`, so strictly speaking it should not be the the `ggml` public API header (but this is of course not your fault or the issue handled by the PR).

---

👤 **Thireus** commented on **2025-07-15** at **06:35:54**

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

👤 **saood06** commented on **2025-07-15** at **06:37:41**

> along with a `cmake` variable that can be used to set `GGML_MAX_CONTEXTS`? You can then build the tool suite with whatever number of contexts you like (the way things are going, soon even 2048 may not be enough).

For a dynamic `GGML_MAX_CONTEXTS` can the windows commit I describe can be set according to this limit (capped at 8192), and included?

---

👤 **ikawrakow** commented on **2025-07-15** at **06:44:17**

> For a dynamic GGML_MAX_CONTEXTS can the windows commit I describe can be set according to this limit (capped at 8192), and included?

Don't understand this comment. Which windows commit and when is there dynamic `GGML_MAX_CONTEXTS`?

---

👤 **saood06** commented on **2025-07-15** at **06:58:21**

@ikawrakow 

>Which windows commit 

[Thireus@65dd65c](https://github.com/Thireus/ik_llama.cpp/commit/65dd65c10d2dc24cdddbd6255c3841c6a6c1038c)

>and when is there dynamic GGML_MAX_CONTEXTS?

And dynamic in the sense that if `GGML_MAX_CONTEXTS` is below 512 then nothing needs to be set (as 512 is the default), if built above 8192, set to only 8192 (as 8192 is the Windows hard upper limit [and even this is not guaranteed see [this](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/setmaxstdio?view=msvc-160) for more info]).

---

👤 **Thireus** commented on **2025-07-15** at **08:30:07**

https://github.com/Thireus/ik_llama.cpp/commit/65dd65c10d2dc24cdddbd6255c3841c6a6c1038c would be a separate pull request as this is a different limitation (OS limitation for number of opened files), that code is required for Windows while other platforms (linux, macos) can use ulimit to lift the limitation.

---

👤 **saood06** commented on **2025-07-16** at **00:31:03**

> [Thireus@65dd65c](https://github.com/Thireus/ik_llama.cpp/commit/65dd65c10d2dc24cdddbd6255c3841c6a6c1038c) would be a separate pull request as this is a different limitation (OS limitation for number of opened files), that code is required for Windows while other platforms (linux, macos) can use ulimit to lift the limitation.

Thanks.

---

👤 **ikawrakow** approved this pull request ✅ on **2025-07-16** at **12:11:08**

---

👤 **Thireus** commented on **2025-07-16** at **23:47:10**

@saood06 - https://github.com/ikawrakow/ik_llama.cpp/pull/620