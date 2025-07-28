## 🔀 [Pull Request #504](https://github.com/ikawrakow/ik_llama.cpp/pull/504) - Add DRY and fix the server to use other new samplers.

| **Author** | `Ph0rk0z` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Source Branch** | `main` |
| **Target Branch** | `main` |
| **Created** | 2025-06-07 |
| **Updated** | 2025-06-13 |

---

## 📄 Description

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [ ] Low
  - [X] Medium
  - [ ] High

So with some vibe coding I added what should be a working dry implementation. Nothing has exploded. Also the server was never modified to use the new samplers so they did nothing unless you were using the main llama.cpp executable without a front end.

I didn't update docs as the other PR seems to do that (but not the server, lol). Let me know if this is any good or not. Much lighter than porting the new sampler arch. 

There's also a spot in the header where sampler order array was never updated? Does it have to be?

---

## 💬 Conversation

👤 **saood06** commented on **2025-06-08** at **05:11:07**

I see you also used the Z-algorithm and the implementation looks the same as llama.cpp (which was based on the implementation done in kobold.cpp) but you stripped the comments explaining it and where it came from. 

Any reason for that choice?

---

👤 **Ph0rk0z** commented on **2025-06-08** at **10:24:01**

Nope. If you find a mistake or a place that needs a comment/credit/etc please point it out. All I care is that it works and we have dry.

---

👤 **saood06** commented on **2025-06-08** at **10:55:44**

> Nope. If you find a mistake or a place that needs a comment/credit/etc please point it out.

I did point it out though?

The whole code of `llama_sample_dry_impl` looks like the exact same as `llama_sampler_dry_apply` from the DRY PR of mainline. Link to the PR (and specific file referenced) [here](https://github.com/ggml-org/llama.cpp/pull/9702/files#diff-ccfd27e7598c9965070306d4c6baf3cb4bf844211d1d37d7c52b0d03c8624507) as a reference.

But the difference is it is lacking pretty much all of comments (which contain attributions alongside a lot of helpful info) that are contained in the mainline PR. 

Initially I only looked at the Z-algorithm because I was keeping up with the initial DRY PR in mainline and I knew that it was stalled waiting for permission for that specific code allowed into an MIT project (as kobold.cpp is AGPL-3.0 ) but now I see that what I said applies to that entire function not just the Z-algorithm.

>All I care is that it works and we have dry.

That may be what you care about, but attribution and credit even when not required (I am not sure it is here, but IANAL) is a nice thing to give, and it looks especially bad considering it really does look like you copy and pasted the code and then removed the attributions and comments.

I am not saying that is what you did (I can't know, so I won't assume), but it definitely does look that way considering the code is identical (but the comments are not) and that is not a good look.

---

👤 **ikawrakow** commented on **2025-06-08** at **11:28:21**

I agree with @saood06. Let's not remove the credits and comments.

---

👤 **Ph0rk0z** commented on **2025-06-08** at **11:46:41**

It went through a LLM but you're working up some scenario where I actively went through and took them out. I'll put them back best I can.

---

👤 **saood06** commented on **2025-06-08** at **11:57:42**

> It went through a LLM but you're working up some scenario where I actively went through and took them out. 

I did originally ask more politely "Any reason for that choice?" and you didn't offer an explanation so I wanted to make it clear what it looks like happened, and I even stated "I can't know, so I won't assume" and I was going to even reference you stating you did vibe coding, but the point was that it looks identical to that like that, and that does impact how people perceive it.

Even if you didn't actively take them out (which I believe you when you say you didn't), you did submit a PR where they were stripped out.

---

👤 **Ph0rk0z** commented on **2025-06-08** at **12:04:21**

It doesn't match their code 1:1 copy pasted.. putting the comments in sort of reveals that. Parts of it do. Its an amalgamation of the PR which was built from k.cpp which itself is probably based on pew and textgen webui code.

---

👤 **saood06** commented on **2025-06-08** at **12:28:52**

> It doesn't match their code 1:1 copy pasted.. putting the comments in sort of reveals that. Parts of it do. Its an amalgamation of the PR which was built from k.cpp which itself is probably based on pew and textgen webui code.

Enough of it did where me looking at both side by side made me feel the need to say something. I never stated it was a copy and paste, but unless you look closely it is hard to tell that it isn't.

Thank you for putting in the work to make this PR I do appreciate it, sorry that didn't come across in my earlier comments, but I still stand by what I said (but maybe I should have included the thank you earlier).

---

👤 **saood06** started a conversation on `src/llama-sampling.cpp` on **2025-06-08** at **12:30:47**

Is this correct? And even if it is why subtract one then add one?

> 👤 **Ph0rk0z** replied on **2025-06-08** at **12:43:31**
> 
> It's LLM jank. Model trying to follow the logic of the operation and show it, despite it being mathematically nonsensical.

> 👤 **saood06** replied on **2025-06-08** at **12:58:55**
> 
> Yes, but that still doesn't answer my question of is it correct? It doesn't look equivalent to the reference implementation to me.

---

👤 **saood06** started a conversation on `src/llama-sampling.cpp` on **2025-06-08** at **12:31:14**

You accidentally duplicated this when pasting in the comment.

---

👤 **saood06** commented on **2025-06-08** at **12:32:40**

I haven't built or ran the code yet, don't have time to test it tonight.

---

👤 **Ph0rk0z** commented on **2025-06-08** at **12:53:16**

That's fine, I hope that more people test it than just us. Remember that dry removes/breaks up engrams not single word repetition. I'll pull changes from here back in and keep rolling with it. Also another reminder that anyone using XTC or n sigma on server was not having it apply. The parameters weren't there.

Need to figure out if new samplers all belong here in sampling.h too


```
    std::vector<llama_sampler_type> samplers_sequence = {
        llama_sampler_type::TOP_K,
        llama_sampler_type::TFS_Z,
        llama_sampler_type::TYPICAL_P,
        llama_sampler_type::TOP_P,
        llama_sampler_type::MIN_P,
        llama_sampler_type::TEMPERATURE
    };
```

Edit: this is default sampler order.. so makes no difference if you want no new samplers within it.

---

👤 **saood06** started a conversation on `src/llama-sampling.cpp` on **2025-06-08** at **13:04:40**

This also looks different from the reference implementation but also you never actual use ring_buffer let alone this method even though you do provide an implementation for it.

---

👤 **saood06** started a conversation on `src/llama-sampling.h` on **2025-06-08** at **13:06:56**

The reference uses a ring_buffer for `dry_last_tokens` (`last_tokens` in the reference implementation) and not a vector. You added an implementation for a ring_buffer but never used it. If you want to use a vector (which could work but I feel like would end up being more complicated) for this than remove the ring_buffer implementation, but I do think you should try and get closer to the original implementation as they did use a ring_buffer for a reason.

---

👤 **saood06** commented on **2025-06-08** at **13:09:57**

> It doesn't match their code 1:1 copy pasted

From my experience porting code from mainline it is usually easier to do that and then fix incompatibilities and any other issues than to do what you did. It also makes reviewing it easier.

---

👤 **Ph0rk0z** commented on **2025-06-08** at **13:11:22**

Yea, in this case it is much much too different. I took several cracks at that and failed each time.

---

👤 **saood06** commented on **2025-06-08** at **13:20:16**

> I haven't built or ran the code yet, don't have time to test it tonight.

I did leave some more comments from just reading the code, I don't think it is worth testing anyway until they are resolved.

---

👤 **Ph0rk0z** commented on **2025-06-08** at **13:57:00**

It does compile and work while penalizing tokens per debug messages.

why does it show *comments* as pending in gh?

![comments](https://github.com/user-attachments/assets/29dcd48c-7f63-4789-b5ef-b8f697926369)

---

👤 **saood06** commented on **2025-06-09** at **10:31:04**

> why does it show _comments_ as pending in gh?

That is odd.

If you want I can try to port the ring buffer if you say it offers better efficiency, but I am testing it as it is right now. 

I'll approve or request changes based on that.

---

👤 **Ph0rk0z** commented on **2025-06-09** at **10:38:59**

I tried with the RB and it caused more problems. Unless there's some big slowdowns, its probably not worth it. Another "trick" directly from pew was to set high top_K like (i.e 100) and place it before DRY to speed everything up. I've been doing that on mainline since I heard about it. Here I already did DRY on/off and the t/s was the same. Probably the thing to look out for.