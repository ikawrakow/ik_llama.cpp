### 🔀 [#510](https://github.com/ikawrakow/ik_llama.cpp/pull/510) - Update News section of readme

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-09 |
| **Updated** | 2025-06-13 |

---

#### Description

@ikawrakow 

Making this draft PR to get your feedback on this format before I add all the new ones (and add in all the missing links). 

Do you see any way to condense the sections that are currently one PR per line? (Maybe subsections of Performance improvements?)

And if any of them can be removed as they are no longer relevant (especially if MLA-2 is deprecated)

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-06-09** at **13:20:55**:<br>

Yes, you can split it like this

---

👤 **saood06** commented the **2025-06-11** at **04:54:07**:<br>

@ikawrakow 

I have added in all the new PR's (skipping a few trivial ones).

I still need to add the PR links for the old stuff, but this still feels too long and organization (ordering, categorization, omission/inclusion) feels like it could still be improved.

Any thoughts?

---

👤 **ikawrakow** submitted a review the **2025-06-11** at **05:41:57**: 💬 `COMMENTED`

---

👤 **ikawrakow** commented during a code review the **2025-06-11** at **05:41:57** on `README.md`:<br>

And not GLM-4, LlaMA-4, Qwen3/Qwen3-MoE ?

---

👤 **ikawrakow** commented during a code review the **2025-06-11** at **05:43:24** on `README.md`:<br>

I would count the trellis quants also here. They partially implemented a long time ago, but the PRs to add CPU and Metal support are quite recent.

---

👤 **ikawrakow** commented during a code review the **2025-06-11** at **05:44:57** on `README.md`:<br>

Duplicate

---

👤 **ikawrakow** submitted a review the **2025-06-11** at **05:45:58**: 💬 `COMMENTED`

---

👤 **saood06** submitted a review the **2025-06-11** at **05:49:53**: 💬 `COMMENTED`

---

👤 **saood06** commented during a code review the **2025-06-11** at **05:49:53** on `README.md`:<br>

Not sure what you mean, all three you mentioned are included alongside their respective PRs. (Qwen 3 is just listed as Qwen3 and not Qwen3/Qwen3-MoE)

---

👤 **ikawrakow** submitted a review the **2025-06-11** at **05:52:58**: 💬 `COMMENTED`

---

👤 **ikawrakow** commented during a code review the **2025-06-11** at **05:52:58** on `README.md`:<br>

Oh, sorry, short attention span. Didn't reed the whole line. It seems I need LLM support when reviewing.

---

👤 **saood06** submitted a review the **2025-06-11** at **05:54:05**: 💬 `COMMENTED`

---

👤 **saood06** submitted a review the **2025-06-11** at **05:55:02**: 💬 `COMMENTED`

---

👤 **saood06** commented during a code review the **2025-06-11** at **05:55:02** on `README.md`:<br>

It really isn't entirely your fault. I don't like this being one block but if I split it into multiple lines it takes too much space.

---

👤 **saood06** submitted a review the **2025-06-11** at **06:18:03**: 💬 `COMMENTED`

---

👤 **saood06** commented during a code review the **2025-06-11** at **06:18:03** on `README.md`:<br>

Fixed.

---

👤 **saood06** submitted a review the **2025-06-11** at **06:19:12**: 💬 `COMMENTED`

---

👤 **ikawrakow** submitted a review the **2025-06-11** at **06:55:52**: 💬 `COMMENTED`

---

👤 **ikawrakow** commented during a code review the **2025-06-11** at **06:55:52** on `README.md`:<br>

Sure.

One thing that bothers me is that many people appear to be thinking that they need `ik_llama.cpp`-specific quants to use `ik_llama.cpp`. Or that they need to do something additional in order to be able to use `llama.cpp` GGUFs with `ik_llama.cpp`. At least this is the impression I get from the comments people make here. I think it would be useful to point out that they can grab any GGUF and just use it the way it is will `ik_llama.cpp`.

---

👤 **saood06** submitted a review the **2025-06-11** at **07:12:17**: 💬 `COMMENTED`

---

👤 **saood06** commented the **2025-06-12** at **16:57:29**:<br>

> Will you finish it, or are you waiting for me to finish it?

I was waiting for a response on what to do to help clarify that people can use existing GGUFs (assuming model support exists here). I just added the missing PR links and am doing the IQK quants section now.

Overall although I think this is an improvement and will be shorter than the old approach as time goes on, I still think it still has the same problem of it will just keep getting longer (and may already be too long).

---

👤 **ikawrakow** submitted a review the **2025-06-13** at **04:56:31**: ✅ `APPROVED`