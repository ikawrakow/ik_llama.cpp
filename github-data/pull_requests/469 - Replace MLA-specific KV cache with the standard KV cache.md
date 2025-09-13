## 🔀 [Pull Request #469](https://github.com/ikawrakow/ik_llama.cpp/pull/469) - Replace MLA-specific KV cache with the standard KV cache

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/remove_kv_l` |
| **Target Branch** | `main` |
| **Created** | 2025-05-28 |
| **Updated** | 2025-05-30 |
| **Merged** | 2025-05-30 |

---

## 📄 Description

Also tried handling the case of a missing V cache (as it happens with most MLA options) when reading/writing/de-fragmenting the cache, but not sure of that works, so making the PR a draft.

---

## 💬 Conversation

👤 **saood06** commented on **2025-05-29** at **05:01:59**

I'll try to test this later tonight (my server is currently busy downloading and converting the new R1 checkpoint) with some loading and saving of the cache to a file but I don't see how de-fragmenting has changed looking at your commits. 

De-fragmenting the cache is not a feature I'm very familiar with at all so I'm not sure how to test/trigger it easily.

---

👤 **ikawrakow** commented on **2025-05-29** at **05:05:56**

> but I don't see how de-fragmenting has changed looking at your commits.

In the function `build_defrag()` there is a check for the presence of V-cache.

---

👤 **saood06** commented on **2025-05-29** at **05:15:41**

> > but I don't see how de-fragmenting has changed looking at your commits.
> 
> In the function `build_defrag()` there is a check for the presence of V-cache.

I see it now. I also see that mainline has changed the default `defrag_thold` (no idea why they call it that when they use _threshold for another variable), so that it is enabled by default but over here it is still disabled by default. Once I familiarize myself with it, I may make a PR that changes the default here.

---

👤 **saood06** commented on **2025-05-29** at **13:00:20**

This still results in a `Segmentation fault` when saving. Similar to before the file it was saving has the initial magic numbers and the number of tokens and what I'm assuming is the raw tokens of the prompt (size mostly lines up missing the last few it seems probably because it didn't flush the buffer before it crashed). So it seems to have crashed again when it attempted to write the actual cache.

---

👤 **ikawrakow** commented on **2025-05-29** at **13:23:10**

Can you debug?

---

👤 **saood06** commented on **2025-05-29** at **13:31:05**

> Can you debug?

I'll look into it more later. Going to head off now, was hoping to have more time for this but downloading and converting the new R1 took a while.

---

👤 **saood06** commented on **2025-05-30** at **06:42:21**

@ikawrakow 

See [#473](https://github.com/ikawrakow/ik_llama.cpp/issues/473) I made that PR to merge onto this branch as I didn't know if you were fine with me pushing commits directly onto your branch.

---

👤 **saood06** commented on **2025-05-30** at **07:57:07**

If you are waiting for me to test de-fragmenting the cache before marking this ready, I'm not sure if/when I will do that, as there doesn't seem to be any indication of when that happens in any example (server only tells you when fragmentation may be an issue). I'd either need to write an example or understand how it works well enough to create a situation in which I know it will happen (with the threshold I set, since as it stands it is disabled by default here).

---

👤 **ikawrakow** commented on **2025-05-30** at **07:59:18**

Closed in favor of [#473](https://github.com/ikawrakow/ik_llama.cpp/issues/473)

---

👤 **saood06** commented on **2025-05-30** at **08:03:29**

@ikawrakow 

[#473](https://github.com/ikawrakow/ik_llama.cpp/issues/473) merged onto `ik/remove_kv_l` and not main, sorry if that wasn't clear before.

---

👤 **ikawrakow** commented on **2025-05-30** at **08:05:17**

Oops.

---

👤 **ikawrakow** commented on **2025-05-30** at **08:08:05**

For the de-fragmentation part of it, to me it looks like it should work. The fact that we didn't get any bug reports so far indicates that the de-fragmentation use case is not very important, so we can just wait and see.