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

#### 🔀 Conversation

👤 **ikawrakow** commented on **2025-05-30** at **06:46:17**

I have missed the double printing of the KV cache size. Do you want to fix it in this PR?

---

👤 **saood06** commented on **2025-05-30** at **06:51:24**

> I have missed the double printing of the KV cache size. Do you want to fix it in this PR?

Sure. I'll fix that and an indentation mistake in the commit I made.

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