### 🐛 [#504](https://github.com/ikawrakow/ik_llama.cpp/pull/504) - Add DRY and fix the server to use other new samplers.

| **Author** | `Ph0rk0z` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-06-07 |
| **Updated** | 2025-06-13 |

---

#### Description

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [ ] Low
  - [X] Medium
  - [ ] High

So with some vibe coding I added what should be a working dry implementation. Nothing has exploded. Also the server was never modified to use the new samplers so they did nothing unless you were using the main llama.cpp executable without a front end.

I didn't update docs as the other PR seems to do that (but not the server, lol). Let me know if this is any good or not. Much lighter than porting the new sampler arch. 

There's also a spot in the header where sampler order array was never updated? Does it have to be?

---

#### 💬 Conversation

👤 **saood06** commented the **2025-06-08** at **05:11:07**:<br>

I see you also used the Z-algorithm and the implementation looks the same but you stripped the comments explaining it and where it came from. Any reason for that choice?

---

👤 **saood06** commented the **2025-06-08** at **10:55:44**:<br>

> Nope. If you find a mistake or a place that needs a comment/credit/etc please point it out.

I did point it out though?

The whole code of `llama_sample_dry_impl` looks like the exact same as `llama_sampler_dry_apply` from the DRY PR of mainline link to the file in question from that PR here https://github.com/ggml-org/llama.cpp/pull/9702/files#diff-ccfd27e7598c9965070306d4c6baf3cb4bf844211d1d37d7c52b0d03c8624507

But the difference is it is lacking pretty much all of comments (which contain attributions alongside a lot of helpful info) that are contained in the mainline PR. 

>All I care is that it works and we have dry.

That may be what you care about, but attribution and credit even when not required (I am not sure it is here, but IANAL) is a nice thing to give, and it looks especially bad considering it really does look like you copy and pasted the code and then removed the attributions and comments.

I am not saying that is what you did (I can't know, so I won't assume), but it definitely does look that way considering the code is identical and that is not a good look.

---

👤 **ikawrakow** commented the **2025-06-08** at **11:28:21**:<br>

I agree with @saood06. Let's not remove the credits and comments.

---

👤 **Ph0rk0z** commented the **2025-06-08** at **11:46:41**:<br>

It went through a LLM but you're working up some scenario where I actively went through and took them out. I'll put them back best I can.

---

👤 **saood06** commented the **2025-06-08** at **11:57:42**:<br>

> It went through a LLM but you're working up some scenario where I actively went through and took them out. 

I did originally ask more politely "Any reason for that choice?" and you didn't offer an explanation so I wanted to make it clear what it looks like happened, and I even stated "I can't know, so I won't assume" and I was going to even reference you stating you did vibe coding, but the point was that it looks identical to that like that, and that does impact how people perceive it.

Even if you didn't actively take them out (which I believe you when you say you didn't), you did submit a PR where they were stripped out.

---

👤 **saood06** commented the **2025-06-08** at **12:28:52**:<br>

> It doesn't match their code 1:1 copy pasted.. putting the comments in sort of reveals that. Parts of it do. Its an amalgamation of the PR which was built from k.cpp which itself is probably based on pew and textgen webui code.

Enough of it did where me looking at both side by side made me feel the need to say something. I never stated it was a copy and paste, but unless you look closely it is hard to tell that it isn't.

Thank you for putting in the work to make this PR I do appreciate it, sorry that didn't come across in my earlier comments, but I still stand by what I said (but maybe I should have included the thank you earlier).

---

👤 **saood06** submitted a review the **2025-06-08** at **12:30:47**: 💬 `COMMENTED`

---

👤 **saood06** commented during a code review the **2025-06-08** at **12:30:47** on `src/llama-sampling.cpp`:<br>

Is this correct? And even if it is why subtract one then add it?

---

👤 **saood06** submitted a review the **2025-06-08** at **12:31:14**: 💬 `COMMENTED`

---

👤 **saood06** commented during a code review the **2025-06-08** at **12:31:14** on `src/llama-sampling.cpp`:<br>

You accidentally duplicated this when pasting in the comment.

---

👤 **Ph0rk0z** submitted a review the **2025-06-08** at **12:43:31**: 💬 `COMMENTED`

---

👤 **Ph0rk0z** commented the **2025-06-08** at **12:53:16**:<br>

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

---

👤 **saood06** submitted a review the **2025-06-08** at **12:58:55**: 💬 `COMMENTED`

---

👤 **saood06** commented during a code review the **2025-06-08** at **12:58:55** on `src/llama-sampling.cpp`:<br>

Yes, but that still doesn't answer my question of is it correct? It doesn't look equivalent to the reference implementation to me.

---

👤 **saood06** submitted a review the **2025-06-08** at **13:04:40**: 💬 `COMMENTED`

---

👤 **saood06** submitted a review the **2025-06-08** at **13:06:56**: 💬 `COMMENTED`

---

👤 **saood06** commented during a code review the **2025-06-08** at **13:06:56** on `src/llama-sampling.h`:<br>

The reference uses a ring_buffer for this and not a vector. You added an implementation for a ring_buffer but never used it

---

👤 **saood06** commented the **2025-06-08** at **13:09:57**:<br>

> It doesn't match their code 1:1 copy pasted

From my experience porting code from mainline it is usually easier to do that and then fix incompatibilities and any other issues than to do what you did. It also makes reviewing it easier.

---

👤 **Ph0rk0z** commented the **2025-06-08** at **13:11:22**:<br>

Yea, in this case it is much much too different. I took several cracks at that and failed each time.

---

👤 **saood06** commented the **2025-06-08** at **13:20:16**:<br>

> I haven't built or ran the code yet, don't have time to test it tonight.

I did leave some more comments though just from reading the code, I don't think it is worth testing anyway until they are resolved.

---

👤 **saood06** commented the **2025-06-09** at **10:31:04**:<br>

> why does it show _comments_ as pending in gh?

That is odd.

If you want I can try to port the ring buffer if you say it offers better efficiency, but I am testing it as it is right now. 

I'll approve or request changes based on that.

---

👤 **Ph0rk0z** commented the **2025-06-09** at **10:38:59**:<br>

I tried with the RB and it caused more problems. Unless there's some big slowdowns, its probably not worth it. Another "trick" directly from pew was to set high top_K like (i.e 100) and place it before DRY to speed everything up. I've been doing that on mainline since I heard about it. Here I already did DRY on/off and the t/s was the same. Probably the thing to look out for.