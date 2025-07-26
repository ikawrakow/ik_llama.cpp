### [Pull Request #473](https://github.com/ikawrakow/ik_llama.cpp/pull/473) - Replace MLA-specific KV cache with the standard KV cache V2

| **Author** | `saood06` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `s6/remove_kv_l` |
| **Target Branch** | `ik/remove_kv_l` |
| **Created** | 2025-05-30 |
| **Updated** | 2025-05-30 |
| **Merged** | 2025-05-30 |

---

#### Description

Tested and was able to successfully read and write the cache to a file. De-fragmenting the cache still has yet to be tested.

It does currently does list the KV size twice (see below), and this seems like a minor regression to me but wanted to ask before I changed it.
```
llama_new_context_with_model: KV self size  = 5369.91 MiB, K (f16): 5369.91 MiB, V (f16):    0.00 MiB
llama_new_context_with_model: KV self size  = 5369.91 MiB, c^KV (f16): 5369.91 MiB, kv^T: not used
```

---

#### 💬 Conversation

👤 **ikawrakow** submitted a review: ✅ `APPROVED` on **2025-05-30** at **06:45:10**

_No content provided._

---

👤 **ikawrakow** commented on **2025-05-30** at **06:46:17**

I have missed the double printing of the KV cache size. Do you want to fix it in this PR?

---

👤 **saood06** commented on **2025-05-30** at **06:51:24**

> I have missed the double printing of the KV cache size. Do you want to fix it in this PR?

Sure. I'll fix that and an indentation mistake in the commit I made.

---

👤 **ikawrakow** submitted a review: ✅ `APPROVED` on **2025-05-30** at **07:28:18**

_No content provided._

---

👤 **saood06** commented on **2025-05-30** at **07:30:43**

Can you just confirm that there is no V-cache for all modes of MLA when flash attention is enabled? I never used type 2 and an earlier PR (#246) says that even without flash attention it doesn't have a V-cache which seems wrong to me.

---

👤 **ikawrakow** commented on **2025-05-30** at **07:35:47**

There is V cache with MLA=1, no FA. In that case the V portion of K gets transposed and stored in the V cache.

---

👤 **saood06** commented on **2025-05-30** at **07:43:49**

> There is V cache with MLA=1, no FA. In that case the V portion of K gets transposed and stored in the V cache.

I understand that, and the code I commited makes the assumption that flash attention plus MLA means no V-cache, MLA without flash attention has a V-cache but still gets printed differently as it is the latent representation of the cache (thus `c^KV`).

I was mostly asking about this:

>mla = 2, fa = 0: FlashMLA . Works only on the CPU and on CUDA. Only small K cache required (the transposed V cache is computed on the fly)

in the linked PR which seems like a typo.

---

👤 **ikawrakow** commented on **2025-05-30** at **08:01:39**

MLA=2 has no V cache with or without FA.

---

👤 **saood06** commented on **2025-05-30** at **08:06:51**

> MLA=2 has no V cache with or without FA.

Do you mind fixing that then, since I wrongfully assumed MLA+FA meant no V-cache.

---

👤 **saood06** commented during a code review on `src/llama.cpp` on **2025-05-30** at **15:24:23**

Given what you said about MLA=2, I don't think this holds. Instead of updating this, I do think passing both would be better even though it is technically a breaking change (unlike previous one which was backwards compatible).

---

👤 **saood06** submitted a review: 💬 `COMMENTED` on **2025-05-30** at **15:24:23**

_No content provided._

---

👤 **ikawrakow** commented during a code review on `src/llama.cpp` on **2025-05-30** at **15:56:29**

Or we simply deprecate MLA=2. The only purpose of it was to have faster prompt processing on CUDA without needing a V cache. Now that there is a FA kernel for head sizes 576,512 also on CUDA, there is basically no point in having MLA=2. I also see many people still using it, which means they are getting lower TG performance.

---

👤 **ikawrakow** submitted a review: 💬 `COMMENTED` on **2025-05-30** at **15:56:29**

_No content provided._

---

👤 **saood06** commented during a code review on `src/llama.cpp` on **2025-05-30** at **16:03:41**

>Or we simply deprecate MLA=2.

Why is MLA=1 being kept? Is there any reason not to use MLA=3? So why not just make MLA a toggle again.

---

👤 **saood06** submitted a review: 💬 `COMMENTED` on **2025-05-30** at **16:03:41**

_No content provided._

---

👤 **ikawrakow** commented during a code review on `src/llama.cpp` on **2025-05-30** at **16:20:40**

> Why is MLA=1 being kept? 

Good question. Mainly to be able to run in the same way as mainline, I guess.

---

👤 **ikawrakow** submitted a review: 💬 `COMMENTED` on **2025-05-30** at **16:20:40**

_No content provided._

---

👤 **ikawrakow** commented during a code review on `src/llama.cpp` on **2025-05-30** at **16:25:20**

MLA=3 has the disadvantage that one needs an additional compute buffer that can become quite large for a long context and a large u-batch size. This can be mitigated with `-amb`, but if one is really operating on the limits of available RAM/VRAM, one may swallow the lower prompt processing performance and use MLA=1 (and for short contexts there isn't much of a difference between MLA=1 and MLA=3)

---

👤 **ikawrakow** submitted a review: 💬 `COMMENTED` on **2025-05-30** at **16:25:20**

_No content provided._

---

👤 **saood06** commented during a code review on `src/llama.cpp` on **2025-05-30** at **16:25:54**

> Mainly to be able to run in the same way as mainline, I guess.

If that is now the main motivation, it might make sense to move it behind a compatibility flag since MLA=3 is such a sane default.

---

👤 **saood06** submitted a review: 💬 `COMMENTED` on **2025-05-30** at **16:25:54**

_No content provided._

---

👤 **saood06** commented during a code review on `src/llama.cpp` on **2025-05-30** at **16:28:30**

> MLA=3 has the disadvantage that one needs an additional compute buffer that can become quite large for a long context and a large u-batch size. This can be mitigated with `-amb`, but if one is really operating on the limits of available RAM/VRAM, one may swallow the lower prompt processing performance and use MLA=1 (and for short contexts there isn't much of a difference between MLA=1 and MLA=3)

That makes sense then maybe a memory optimized flag not compatibility?

---

👤 **saood06** submitted a review: 💬 `COMMENTED` on **2025-05-30** at **16:28:30**

_No content provided._

---

👤 **ikawrakow** commented during a code review on `src/llama.cpp` on **2025-05-30** at **16:34:16**

`-mla fast` and `-mla mem` ?

---

👤 **ikawrakow** submitted a review: 💬 `COMMENTED` on **2025-05-30** at **16:34:16**

_No content provided._

---

👤 **saood06** commented during a code review on `src/llama.cpp` on **2025-05-30** at **17:06:07**

> `-mla fast` and `-mla mem` ?

That sounds good.

---

👤 **saood06** submitted a review: 💬 `COMMENTED` on **2025-05-30** at **17:06:07**

_No content provided._