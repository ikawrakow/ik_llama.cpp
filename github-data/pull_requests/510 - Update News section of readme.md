## 🔀 [Pull Request #510](https://github.com/ikawrakow/ik_llama.cpp/pull/510) - Update News section of readme

| **Author** | `saood06` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `s6/readme_update` |
| **Target Branch** | `main` |
| **Created** | 2025-06-09 |
| **Updated** | 2025-06-13 |
| **Merged** | 2025-06-13 |

---

## 📄 Description

@ikawrakow 

Making this draft PR to get your feedback on this format before I add all the new ones (and add in all the missing links). 

Do you see any way to condense the sections that are currently one PR per line? (Maybe subsections of Performance improvements?)

And if any of them can be removed as they are no longer relevant (especially if MLA-2 is deprecated)

---

## 💬 Conversation

👤 **ikawrakow** commented on **2025-06-09** at **13:20:55**

Yes, you can split it like this

---

👤 **saood06** commented on **2025-06-11** at **04:54:07**

@ikawrakow 

I have added in all the new PR's (skipping a few trivial ones).

I still need to add the PR links for the old stuff, but this still feels too long and organization (ordering, categorization, omission/inclusion) feels like it could still be improved.

Any thoughts?

---

👤 **ikawrakow** started a conversation on `README.md` on **2025-06-11** at **05:41:57**

And not GLM-4, LlaMA-4, Qwen3/Qwen3-MoE ?

> 👤 **saood06** replied on **2025-06-11** at **05:49:53**
> 
> Not sure what you mean, all three you mentioned are included alongside their respective PRs. (Qwen 3 is just listed as Qwen3 and not Qwen3/Qwen3-MoE)

> 👤 **ikawrakow** replied on **2025-06-11** at **05:52:58**
> 
> Oh, sorry, short attention span. Didn't reed the whole line. It seems I need LLM support when reviewing.

> 👤 **saood06** replied on **2025-06-11** at **05:55:02**
> 
> It really isn't entirely your fault. I don't like this being one block but if I split it into multiple lines it takes too much space.

---

👤 **ikawrakow** started a conversation on `README.md` on **2025-06-11** at **05:43:24**

I would count the trellis quants also here. They partially implemented a long time ago, but the PRs to add CPU and Metal support are quite recent.

> 👤 **saood06** replied on **2025-06-11** at **05:54:05**
> 
> That makes sense, I'll remove the line about "Additional implementations for the trellis quants..." and add all of them here alongside the initial CUDA implementation in the closed PR (even though technically it came with the CPU implementation PR).

> 👤 **saood06** replied on **2025-06-11** at **06:19:12**
> 
> I split the Quantization additions section into Trellis and IQK sections.
> 
> This way all the IQK quants can be included (as they are still relevant and people still ask about them) if you think this makes sense.
> 
> Thoughts?

> 👤 **ikawrakow** replied on **2025-06-11** at **06:55:52**
> 
> Sure.
> 
> One thing that bothers me is that many people appear to be thinking that they need `ik_llama.cpp`-specific quants to use `ik_llama.cpp`. Or that they need to do something additional in order to be able to use `llama.cpp` GGUFs with `ik_llama.cpp`. At least this is the impression I get from the comments people make here. I think it would be useful to point out that they can grab any GGUF and just use it the way it is will `ik_llama.cpp`.

> 👤 **saood06** replied on **2025-06-11** at **07:12:16**
> 
> > Sure.
> 
> Okay will add all the IQK quants (and their associated PRs) and also link the discussions where you originally talk about them preceding that. 
> 
> > One thing that bothers me is that many people appear to be thinking that they need `ik_llama.cpp`-specific quants to use `ik_llama.cpp`. Or that they need to do something additional in order to be able to use `llama.cpp` GGUFs with `ik_llama.cpp`. At least this is the impression I get from the comments people make here. 
> 
> Yes, I'm fairly certain I've seen the comments that gave you that impression (and it was a question that was asked of me many times on other platforms).
> 
> >I think it would be useful to point out that they can grab any GGUF and just use it the way it is will `ik_llama.cpp`.
> 
> I agree that could be useful (if done properly), but I don't think that should be handled in the News section. 
> 
> Maybe it should be added to the TL;DR? But I don't know how to rewrite that to include that (and update it in general to something that is more current and clear).

---

👤 **ikawrakow** started a conversation on `README.md` on **2025-06-11** at **05:44:57**

Duplicate

> 👤 **saood06** replied on **2025-06-11** at **06:18:03**
> 
> Fixed.

---

👤 **ikawrakow** commented on **2025-06-12** at **16:26:10**

Will you finish it, or are you waiting for me to finish it?

---

👤 **saood06** commented on **2025-06-12** at **16:57:29**

> Will you finish it, or are you waiting for me to finish it?

I was waiting for a response on what to do to help clarify that people can use existing GGUFs (assuming model support exists here). I just added the missing PR links and am doing the IQK quants section now.

Overall although I think this is an improvement and will be shorter than the old approach as time goes on, I still think it still has the same problem of it will just keep getting longer (and may already be too long).

---

👤 **saood06** commented on **2025-06-12** at **17:30:39**

I marked it ready for review as all the things that needed to be added are now added. I still think it could be better, but I don't have any ideas on how to make it better anymore.

---

👤 **ikawrakow** approved this pull request ✅ on **2025-06-13** at **04:56:31**