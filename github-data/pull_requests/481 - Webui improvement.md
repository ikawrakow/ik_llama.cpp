## 🔀 [Pull Request #481](https://github.com/ikawrakow/ik_llama.cpp/pull/481) - Webui improvement

| **Author** | `firecoperana` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `webui_improvement` |
| **Target Branch** | `main` |
| **Created** | 2025-06-01 |
| **Updated** | 2025-06-10 |
| **Merged** | 2025-06-08 |

---

## 📄 Description

Updating webui to a newer version, but not latest version
Some minor bug fix for webui
- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [ ] Low
  - [x] Medium
  - [ ] High

---

## 💬 Conversation

👤 **ikawrakow** commented on **2025-06-01** at **05:41:30**

I need people to confirm that this works.

---

👤 **saood06** commented on **2025-06-01** at **06:26:56**

I see options for DRY and XTC. Neither of which is currently supported here.

---

👤 **ikawrakow** commented on **2025-06-01** at **06:50:59**

> I see options for DRY and XTC. Neither of which is currently supported here.

Yes, I would have thought that one needs to pick up the changes in `common` before changing the server/WebUI/RPC. But picking up changes in `common` requires picking up changes in `llama`. But picking up changes in `llama` requires picking up changes in `ggml`. But picking up changes in `ggml` requires basically starting fresh and applying the hundreds of changes that I have done to `ggml`. But if I ever considered doing that, then it would be better to actually write my own...

---

👤 **saood06** commented on **2025-06-01** at **07:13:29**

> > I see options for DRY and XTC. Neither of which is currently supported here.
> 
> Yes, I would have thought that one needs to pick up the changes in `common` before changing the server/WebUI/RPC. But picking up changes in `common` requires picking up changes in `llama`. But picking up changes in `llama` requires picking up changes in `ggml`. But picking up changes in `ggml` requires basically starting fresh and applying the hundreds of changes that I have done to `ggml`. But if I ever considered doing that, then it would be better to actually write my own...

Are you sure bringing over samplers is that difficult? There was a time when I wanted to bring over DRY ( I no longer care, min_p and temperature is all I use and n-sigma is the only one that if brought over I may end up using since it might be better at eliminating "bad" tokens at the tail than min_p is, but min_p works well enough that I doubt it would be that big of an improvement), and I looked into it, and the only major issue was that you would have to manually port it over because of the refactors that mainline has done, but it still seemed manageable, and much easier than starting from scratch.

Edit: I want to clarify I saw DRY and XTC from the code. I haven't tested the new Webui.

---

👤 **ikawrakow** commented on **2025-06-01** at **07:32:23**

Adding a sampler or two shouldn't be too hard. But
* This PR is a 12 kLOC change, so possibly being dependent on various other changes in `common` (or even `llama`?) to function correctly (I haven't checked, just guessing).
* I dislike the way sampling is done here. It would be better to adopt the mainline idea of having arbitrary sampling chains that you can stick together any way you like. But try copying `llama-sampling.h` and `llama-sampling.cpp` from mainline and see what happens. So, as you say, it has to be added manually to the existing sampling mechanism that I don't like.

---

👤 **saood06** commented on **2025-06-01** at **08:05:34**

>Adding a sampler or two shouldn't be too hard. But 
>[...]
>I dislike the way sampling is done here. It would be better to adopt the mainline idea of having arbitrary sampling chains that you can stick together any way you like. But try copying `llama-sampling.h` and `llama-sampling.cpp` from mainline and see what happens. So, as you say, it has to be added manually to the existing sampling mechanism that I don't like.

I followed the sampling changes (and new sampler additions) as they were happening and I do agree that it changed for the better, but it does seem like considerably more work to adapt the changes in sampling than just porting over samplers. My own desires made me consider the easier change of just bringing over any sampler I cared enough about (which currently is none), over changing the way sampling is done, but I know that will differ for everyone.

>This PR is a 12 kLOC change, so possibly being dependent on various other changes in common (or even llama?) to function correctly (I haven't checked, just guessing).

I haven't checked either. I only looked through the code so far for this PR (and the RPC one)

---

👤 **Ph0rk0z** commented on **2025-06-01** at **11:52:48**

XTC is about the only way to remove top tokens which could be slop or refusals. Dry has it's issues, but is better than the other repeat penalties. min_p and temperature are fine for non creative stuff but otherwise they come up short. And no "just raise the temperature" isn't a solution.

---

👤 **saood06** commented on **2025-06-01** at **12:38:48**

> XTC is about the only way to remove top tokens which could be slop or refusals. 

XTC is one of the only samplers that is not monotonic (and for a reason, it doesn't really make sense to alter the rankings of the predicted tokens, since so much effort was made training the LLM to rank them in the order it did). I do think that Top-nσ with higher temperatures is better for diverse branching over using XTC but that is mostly just based on the math behind them. I don't get using a sampler to remove refusals either use a model that doesn't refuse, or prefill some of the response so that it doesn't refuse.

>Dry has it's issues, but is better than the other repeat penalties.

I agree, but like you said and from what I've heard it still has it's issues, and so manually intervening to fix repeats is still better as that doesn't have issues.

>min_p and temperature are fine for non creative stuff but otherwise they come up short. And no "just raise the temperature" isn't a solution.

I disagree, min_p does fine at removing the "bad" tail end, and temperature works for regulating how "chaotic" a model is, and that is all you need (maybe Top-nσ over min_p as it may be better at removing the "bad" tail end at higher temperatures). I do often look at the top-10 tokens and manually sample or even inject tokens to steer the output, thus "manually" sampling, but even without that from what I can see from all the token distributions I've looked at, temperature and min_p leave little room for improvement.

---

👤 **Ph0rk0z** commented on **2025-06-01** at **13:47:32**

> since so much effort was made training the LLM to rank them in the order it did

Right, and I want to undo it. Trainers and my goals aren't necessarily aligned. 

>either use a model that doesn't refuse, or prefill some of the response so that it doesn't refuse.

First part doesn't exist. Second part is already done. Models like to maliciously comply or default to cliches. Dumping top tokens goes a long way.

>I disagree, min_p does fine at removing the "bad" tail end

Yes it does, as well as setting high top_K like 100. I use min_P of around .03 on everything. But cranking the temperature doesn't really improve *coherent* creativity. It just makes the model chaotic.

>manually sample or even inject tokens to steer the output, thus "manually" sampling
>manually intervening to fix repeats

Absolutely kills the fun for me. We're coming at it from 2 different places. I want a realistic "personality" with no defined end goal. A chat videogame. You probably want a story that goes somewhere you have planned it to go.

In either case, taking the sampling refactor from mainline probably does it all at once. It didn't look super easy from the PRs unfortunately. They did a lot of changes. Even when I was trying to add tensor size printing, everything is all renamed or moved. IK not kidding about how they do that constantly.

---

👤 **saood06** commented on **2025-06-01** at **14:40:33**

> > since so much effort was made training the LLM to rank them in the order it did
> 
> Right, and I want to undo it. Trainers and my goals aren't necessarily aligned.

Fair enough.

> > either use a model that doesn't refuse, or prefill some of the response so that it doesn't refuse.
> 
> First part doesn't exist. Second part is already done. Models like to maliciously comply or default to cliches. Dumping top tokens goes a long way.

I'll take your word for it, since the models I prefer to use now have basically never given me a refusal, and the ones I used to use that would sometimes refuse, the prefilling did work. I think I do remember what you are referring to happening and I would usually just not use those models for those tasks.

>But cranking the temperature doesn't really improve _coherent_ creativity. It just makes the model chaotic.

Have you tried Top-nσ since it is designed to maintain coherence while acting similar to min_p at high temperatures. I've read mixed feedback from people, but personally I prefer lower temperatures (if the model works well with it, which is why I liked that the new V3 recommended 0.3 which I use with min_p of 0.01, but other models don't work as well with such low temperatures and I would often use 0.6-1.2 depending on the model).

> > manually sample or even inject tokens to steer the output, thus "manually" sampling
> > manually intervening to fix repeats
> 
> Absolutely kills the fun for me. We're coming at it from 2 different places. I want a realistic "personality" with no defined end goal. A chat videogame. You probably want a story that goes somewhere you have planned it to go.

I just realized repeat loops haven't happened for me in a long time, but fixing them was definitely not fun. Even if I don't steer, seeing the top-10 tokens is interesting to me. A story writing assistant is one of the ways I use LLMs but it definitely isn't the only way I use them.

You are correct that I haven't use them for what you call "a chat videogame" but I definitely wouldn't be opposed to it, I just haven't written a prompt that sets that up (or used one written by someone else), and I can understand why in that situation intervening or injecting tokens could be very annoying.

We probably do use different front-ends then as well. I mainly use (and have contributed to) mikupad, but if I were to try what you describe I know there are other front-ends that would work better.

> In either case, taking the sampling refactor from mainline probably does it all at once. It didn't look super easy from the PRs unfortunately. They did a lot of changes. Even when I was trying to add tensor size printing, everything is all renamed or moved. IK not kidding about how they do that constantly.

Yeah, it doesn't look easy, I didn't look into it with the purpose of bringing it over, but I have looked at all basically all of those PRs and the code and I do agree that bringing it over would be a good amount of work.

---

👤 **Ph0rk0z** commented on **2025-06-01** at **18:06:57**

Have not tried top n sigma since it's only in mainline and generally I use EXL2 for normal sized models. I've been meaning to load up command-A or gemma and give it a whirl. All the "meme" sampling missing here is a bit of a drawback. I initially didn't even realize that it was forked pre dry/xtc and was confused why Deepseek 2.5 was looping so badly. Its like you have to choose between usable speed (close to fully offloaded dense model) or functionality.

---

👤 **ikawrakow** commented on **2025-06-02** at **06:07:35**

> Its like you have to choose between usable speed (close to fully offloaded dense model) or functionality

Interesting take. Isn't usable speed one of the most important functionalities of an LLM inference toolkit?

---

👤 **ikawrakow** commented on **2025-06-02** at **09:24:33**

No user feedback here, so new strategy: I'll merge this tomorrow. If we don't get bug reports, all is good. If we do get bug reports, all is good too because we know that it needs further work.

---

👤 **saood06** commented on **2025-06-02** at **10:25:34**

> No user feedback here, so new strategy: I'll merge this tomorrow. If we don't get bug reports, all is good. If we do get bug reports, all is good too because we know that it needs further work.

The DRY/XTC options in the UI this adds can't function. I don't think there is a need to test that, those samplers do not exist here, so the UI exposing them should be removed before this is added (or the samplers could be added I guess).

The other thing I found when looking at the source code is the bug report button goes to mainline and not here.

---

👤 **Ph0rk0z** commented on **2025-06-02** at **11:21:04**

> Isn't usable speed one of the most important functionalities of an LLM inference toolkit?

Right.

But then god said:

>Deepseek 2.5 was looping so badly

So it's making me badly want to port the QOL stuff. It mirrors LLMs where a model will be great and then has that one thing you want to change.

---

👤 **ikawrakow** commented on **2025-06-02** at **12:53:48**

> So it's making me badly want to port the QOL stuff. It mirrors LLMs where a model will be great and then has that one thing you want to change.

I would love that, and I'm sure many users will too.

---

👤 **Ph0rk0z** commented on **2025-06-02** at **15:29:07**

Ok.. well it seemed easy enough until I hit the portion where they refactored everything into args.h/args.cpp. So all those new things you added aren't in ctx params anymore. Some time around September. Looks fun, doesn't it? https://github.com/ggml-org/llama.cpp/commit/bfe76d4a17228bfd1565761f203123bc4914771b

---

👤 **ikawrakow** commented on **2025-06-03** at **06:34:03**

@Ph0rk0z See [#486](https://github.com/ikawrakow/ik_llama.cpp/issues/486) for the XTC sampler

---

👤 **Ph0rk0z** commented on **2025-06-03** at **11:27:29**

Ha! Last night I cherry picked and got the refactor working. Got as far as DRY and XTC. I didn't post it yet because I somehow bugged the seed to where it it might not be randomizing on re-rolls. I was gonna keep going after a night of sleep. Adding sigma was good because its way up there, past yet another refactor.

---

👤 **Ph0rk0z** commented on **2025-06-03** at **11:53:08**

https://github.com/Ph0rk0z/ik_llama.cpp/branches

Btw, there is a branch where it's only refactored to separate out the sampling. Furthest ahead one is the DRY one. Still didn't delete the args.cpp nor fixed the Makefile changes mainline did but you get the gist. Is any of that worth doing?

---

👤 **ikawrakow** commented on **2025-06-03** at **12:27:30**

Too much change for my taste. The DRY one is 8631+ LOC, 4089- LOC. The XTC one is 7687+, 4020-. This would require a lot of testing. My PR's are in the 70-90 LOC each. The DRY would be a bit bigger, but not sure if it is worth it.

---

👤 **Ph0rk0z** commented on **2025-06-03** at **13:25:06**

Yep, it is a ton of changes. They add a lot of code in a year. I'm surprised it worked at all. Much of it is related to all the examples too. Even here, 60 files changed for the webui.

---

👤 **pt13762104** commented on **2025-06-05** at **02:39:22**

Clicking the save button in settings doesn't exit it out like llama.cpp

---

👤 **ikawrakow** commented on **2025-06-05** at **06:44:33**

> Clicking the save button in settings doesn't exit it out like llama.cpp

Thanks for testing. Apart from this, does it work for you?

---

👤 **pt13762104** commented on **2025-06-05** at **14:34:51**

It works... At least I found it can respond properly and show TPS. Might need more testing.

---

👤 **firecoperana** commented on **2025-06-07** at **23:02:59**

> Clicking the save button in settings doesn't exit it out like llama.cpp

I think the issue is because you used the newest version of webui from mainline in the same browser. If you click "reset to default", save is working again.

---

👤 **pt13762104** commented on **2025-06-08** at **02:07:43**

I'll try, thanks

---

👤 **saood06** commented on **2025-06-08** at **05:02:29**

@firecoperana 

If you are interested I added a new endpoint to server that could be utilized by this front end ([#502](https://github.com/ikawrakow/ik_llama.cpp/issues/502)). I already added support to my preferred front end and it has been nice being able to see all my stored sessions and restore them with ease (saving and restoring support already existed but there was no good way to add it to a UI without being able to list what is saved which is what I added).

---

👤 **iehgit** commented on **2025-06-08** at **08:04:31**

Works fine (multiple conversations, display of token rate). Huge improvement over the old UI, which made you choose between prompt formats that didn't fit to current models.

---

👤 **firecoperana** commented on **2025-06-08** at **15:21:03**

> @firecoperana
> 
> If you are interested I added a new endpoint to server that could be utilized by this front end ([#502](https://github.com/ikawrakow/ik_llama.cpp/issues/502)). I already added support to my preferred front end and it has been nice being able to see all my stored sessions and restore them with ease (saving and restoring support already existed but there was no good way to add it to a UI without being able to list what is saved which is what I added).

I will try when I have time. That looks very helpful!

---

👤 **saood06** commented on **2025-06-09** at **09:23:32**

@ikawrakow 

What is your opinion on having another additional (alternative like legacy) frontend besides the one implemented here. The one I use has what seems like an abandoned maintainer so I have nowhere to upstream my changes.

---

👤 **ikawrakow** commented on **2025-06-09** at **10:22:37**

So you want to bring in to this repository your favorite frontend and maintain it here?

---

👤 **saood06** commented on **2025-06-09** at **10:39:34**

> So you want to bring in to this repository your favorite frontend and maintain it here?

Yes.

---

👤 **ikawrakow** commented on **2025-06-09** at **10:45:37**

Can I take a look?

---

👤 **saood06** commented on **2025-06-09** at **10:51:01**

> Can I take a look?

For now what is public is https://github.com/lmg-anon/mikupad/pull/113. But I have more that isn't public as it works but is not polished (like adding [#502](https://github.com/ikawrakow/ik_llama.cpp/issues/502) and [#504](https://github.com/ikawrakow/ik_llama.cpp/issues/504) ) and other things in the roadmap.

---

👤 **ikawrakow** commented on **2025-06-09** at **11:07:34**

It doesn't look like a very big project, so from that point of view, sure.

But what about license and such?

Why do you prefer do have it here instead of just a separate fork?

---

👤 **saood06** commented on **2025-06-09** at **11:19:25**

> It doesn't look like a very big project, so from that point of view, sure 
> But what about license and such?

It has a very permissible license, which allows for it to be here from how I read it. ( https://github.com/lmg-anon/mikupad/blob/main/LICENSE )
 
> Why do you prefer do have it here instead of just a separate fork?

I plan to maintain it following the feature support here, and there are changes that would make it integrate better here that I am planning.

---

👤 **ikawrakow** commented on **2025-06-09** at **11:22:13**

I know CC0 is very permissive. What I don't know is how one mixes it with MIT. I.e., do we need to update the license file and such.

---

👤 **saood06** commented on **2025-06-09** at **11:29:42**

> I know CC0 is very permissive. What I don't know is how one mixes it with MIT. I.e., do we need to update the license file and such.

I think we can just add a CC0 section to the license file, that specifies the location of it. I could add and maintain an authors file.

---

👤 **ikawrakow** commented on **2025-06-09** at **11:31:36**

OK, go ahead.

---

👤 **saood06** commented on **2025-06-09** at **11:38:39**

> OK, go ahead.

Thanks, I will submit the PR when it is ready.

---

👤 **pt13762104** commented on **2025-06-09** at **13:34:49**

Finally, some decent UI. Now I can ditch openwebui again. I can't just use the old UI, i don't even know where to start. This made my day