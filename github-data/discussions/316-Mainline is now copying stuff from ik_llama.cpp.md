### 🗣️ [#316](https://github.com/ikawrakow/ik_llama.cpp/discussions/316) - Mainline is now copying stuff from ik_llama.cpp

| **Author** | `ikawrakow` |
| :--- | :--- |
| **Created** | 2025-04-06 |
| **Updated** | 2025-04-29 |

---

#### Description

We have [this merged PR](https://github.com/ggml-org/ggml/pull/1174) and [this pending PR](https://github.com/ggml-org/ggml/pull/1179) in the [ggml repository](https://github.com/ggml-org/ggml) copying code from `ik_llama.cpp`. It is an interesting choice of venue. [ggml](https://github.com/ggml-org/ggml) is well known, but much lower profile than [llama.cpp](https://github.com/ggml-org/llama.cpp). We know that changes added to `ggml` quietly make their way into `llama.cpp` with "sync: ggml" PRs such as [this one](https://github.com/ggml-org/llama.cpp/pull/12670).

The merged PR went into `ggml` without attribution (other than the source being mentioned in the PR). The pending PR attributes the change to `<48489457+ikawrakow@users.noreply.github.com>`, so me, but me as one of the (currently) 335 [ggml authors](https://github.com/ggml-org/ggml/blob/master/AUTHORS). But I definitely did not write the code with the intent of contributing it to `ggml`, `llama.cpp`, or any of ggerganov's projects. Does that mean that since I once contributed to `llama.cpp`, the copyright on everything I produce from there on is jointly owned by the 335 `ggml` authors, or perhaps even by the (currently) 1106 [llama.cpp authors](https://github.com/ggml-org/llama.cpp/blob/master/AUTHORS)? 

`ik_llama.cpp` is open source, and it uses the same MIT license as `ggml/llama.cpp`. The MIT license says
```
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

Hmm. The PRs are definitely not a copy of `ik_llama.cpp`, but are they a "substantial portion" of it? How is "substantial" being measured? By LOCs? By utility? By some other measure?

Let's take the [merged PR](https://github.com/ggml-org/ggml/pull/1174). It is just 50 LOC of trivial code. And yet, it does improve prompt processing of `bf16` models by a factor of 2 compared to [this PR](https://github.com/ggml-org/llama.cpp/pull/11093), which added CUDA `bf16` support to `llama.cpp`. The [pending PR](https://github.com/ggml-org/ggml/pull/1179) is just a 69 LOC change of only slightly less trivial code. And yet, it improves PP performance of MoE models with many experts such as DeepSeek-V3/R1/Lite by more than [this 2000+ LOC](https://github.com/ggml-org/llama.cpp/pull/11583) rework of the CUDA matrix multiplication kernels and flash attention implementation. Let's take a look at [this ik_llama.cpp PR](https://github.com/ikawrakow/ik_llama.cpp/pull/307) that has not been discovered yet. The relevant change that improves MoE PP performance is the rewrite of [this kernel](https://github.com/ikawrakow/ik_llama.cpp/blob/ec84855c6ae5a08686f3e5d8010e38064269deb3/ggml/src/ggml-metal.metal#L8541). It is just 60 LOC or so, but the performance gain is many times more than the grand total of all modifications made to the `ggml/llama.cpp` Metal backend since I left these projects in March of 2024. 

So, again, is it utility or number of LOCs that define the copied code as "substantial portion" of the software it was copied from? 

But, hey, IANAL, so it is maybe better to focus on the moral side of things. When I left the `llama.cpp` project, I expressed the wish that all of my contributions be removed. They didn't need to do it legally, but wouldn't it have been nice if they still did? ggerganov cited too much impact on downstream projects. Not on `llama.cpp` itself, but on downstream projects. Because, you know, downstream projects are too inept to add back k-quants, i-wuants, and imatrix after their removal from upstream. In any case, it is known what happened, so it should be obvious to anyone that I don't want my work to be copied into ggerganov's projects. If they were nice, they would have re-implemented these changes - it is not rocket science. And if they were really nice, they would have acknowledged `ik_llama.cpp` for the inspiration. Or, if they didn't feel like re-implementing it, they would add my copyright notice, legally required or not, so we don't need to ponder at what point what they copied became a "substantial portion" of the work they are copying.

---

#### 🗣️ Discussion

👤 **CISC** replied the **2025-04-06** at **13:12:04**:<br>

Uh, I was not aware of any wish for your work to be removed, in fact, I made the PRs solely based on your comment here: https://github.com/ikawrakow/ik_llama.cpp/discussions/256#discussioncomment-12496828

I chose to submit these to `ggml` not for some nefarious reason, but simply because they were restricted to `ggml` code only.

---

👤 **CISC** replied the **2025-04-06** at **13:33:21**:<br>

> Hmm. The PRs are definitely not a copy of `ik_llama.cpp`, but are they a "substantial portion" of it? How is "substantial" being measured? By LOCs? By utility? By some other measure?

TBH I overlooked that you added yourself to the copyright notice, I looked at diffs only. It's simple to fix though, I can add it to any file that has your code merged into it.

> If they were nice, they would have re-implemented these changes - it is not rocket science. And if they were really nice, they would have acknowledged `ik_llama.cpp` for the inspiration. Or, if they didn't feel like re-implementing it, they would add my copyright notice, legally required or not, so we don't need to ponder at what point what they copied became a "substantial portion" of the work they are copying.

Please don't blame anyone else than me, I do not represent `ggml` nor `llama.cpp`, and I acted in good faith.

---

👤 **ikawrakow** replied the **2025-04-06** at **13:50:50**:<br>

@CISC

I'm sorry if this came across as a critique/attack on you. That was not the intent, and it has nothing to do with you. It is between ggerganov and me. Given the history, and there is 15 years of it even before `llama.cpp` came to be, I would have expected a different reaction from ggerganov to your PRs.

> 👤 **JohannesGaessler** replied the **2025-04-06** at **14:06:02**:<br>
> In the end I am the one who is responsible for reviewing and merging the PR in question. I had interpreted [this post](https://github.com/ikawrakow/ik_llama.cpp/discussions/256#discussioncomment-12496828) as permission to do so without preconditions. I'm sorry for acting against your wishes.

---

👤 **CISC** replied the **2025-04-06** at **14:08:38**:<br>

This puts me in a bind though, my intention was to upstream what I could (with the hardware I have available to test) as it seemed you were suggesting that this should be done (but not willing to do yourself).

You have made a great number of awesome contributions here, and I still wish for them to be merged into mainline, as it would improve it greatly, and it might make it simpler for you to rebase and get newer features from mainline as well. This should be a win-win.

---

👤 **ikawrakow** replied the **2025-04-06** at **14:37:07**:<br>

@CISC @JohannesGaessler As you both refer to what I wrote in #256, here it is:

> upstream is free to take from here whatever they find useful

Meaning there is nothing I can do to prevent that from happening as I'm publishing under a MIT license. I don't think I said that I do not expect upstream to abide by the terms of the license.

> 👤 **CISC** replied the **2025-04-06** at **14:38:40**:<br>
> > @CISC @JohannesGaessler As you both refer to what I wrote in #256, here it is:
> > 
> > > upstream is free to take from here whatever they find useful
> > 
> > Meaning there is nothing I can do to prevent that from happening as I'm publishing under a MIT license. I don't think I said that I do not expect upstream to abide by the terms of the license.
> 
> I'm fixing my mistake right now, sorry about that.

---

👤 **ikawrakow** replied the **2025-04-07** at **06:30:56**:<br>

So, this is becoming interesting. Here is what @ggerganov has to say about my copyright notice being included in the file(s) where stuff was copied from my work:

> Including copyright notices is optional since the Berne convention - this was discussed last year: https://github.com/ggml-org/llama.cpp/discussions/6394.
>
> And again - we do provide the notices in the AUTHORS files. There is no need to sprinkle them inside the code.

The [discussion 6934](https://github.com/ggml-org/llama.cpp/discussions/6394) was about Intel engineers copy-pasting CUDA kernels that I wrote into the SYCL implementation and slapping their copyright notice on it (and, to add insult to injury, they were copy-pasting the code into wrong places, and refusing to accept PRs fixing it, which was the actual reason to start the discussion in the first place). The very knowledgable conflict resolution expert with no legal education who came to resolve the conflict said that was OK, because according to the [Berne Convention](https://en.wikipedia.org/wiki/Berne_Convention) they couldn't take away the copyright from me by doing that (I wonder if software was covered in the original Berne Convention agreement of 1886? Just kidding). The copyright is collectively owned by the authors of the project, and their copyright is established by the AUTHORS file, so copyright notices do not need to be present in every file (but apparently it is OK for Intel to have their copyright notice in the file, without further copyright notices).

@ggerganov The work from which it is being copied is not work contributed to your project by me and therefore covered by my name being in the AUTHORS file of your work. Can you please point me to the text in the Berne Convention where it is established that if you copied my work into your work, it would be OK to ignore the terms of the license under which I published my work, and not include  my copyright notice in your work as requested by the MIT license? If you don't like copyright notices "sprinkled inside the code", you have the option to reject the PRs or add my copyright notice to the copyright notice of your project. Oh, another option (if you trust your legal expertise) would be to accept the PRs as is, and then make your own PRs removing the copyright notices. In that way it would be you not being nice to a fellow open source developer with whom you want to "freely exchange ideas" (and possibly violating the terms of their license), not your contributor. I think asking a contributor to do that is going too far. But at the end of the day it is your project, so yes, you can ask your contributors to play by your rules.

---

👤 **JohannesGaessler** replied the **2025-04-07** at **07:59:15**:<br>

For the record: Do you find it acceptable for people to read your code and to then submit a PR to llama.cpp/ggml with the same functionality?

> 👤 **ikawrakow** replied the **2025-04-07** at **09:10:21**:<br>
> > For the record: Do you find it acceptable for people to read your code and to then submit a PR to llama.cpp/ggml with the same functionality?
> 
> I addressed that above. But here it is again my perhaps wrong concept of how it should be:
> * If you copy my code, you need to add a copyright notice as requested by the MIT license.
> * If you reimplement what I have done here in your own way, you don't need to mention me or this repository. But if you were nice, you would still mention the original source/idea. Just like in many places in the ggml/llama.cpp code there are references to papers and/or other repositories. 
> 
> Now, also for the record, it isn't so that there aren't copyright notices in `ggml` "sprinkled around the code" as @ggerganov puts it. See for instance [this](https://github.com/ggml-org/ggml/blob/ab9ed73d40965d7e4b25a4adf2230b9a19bffbf9/src/ggml-cpu/ops.cpp#L4996) (and same notices in all other backends). I have this line in my fork as well in a completely [different place](https://github.com/ikawrakow/ik_llama.cpp/blob/a051f08b8f059fa10dd089d231b975291c122e9d/ggml/src/ggml.c#L16726), so it has been preserved over multiple code reorganizations (so, maintaining copyright notices in the source code as things are moved around is not quite as painful as claimed). You don't wonder why a Kawrakow copyright notice is so different from a Jeffrey Quesnelle and Bowen Peng copyright notice?
> 
> 👤 **JohannesGaessler** replied the **2025-04-07** at **10:41:05**:<br>
> Thank you for your input. My perspective is that I don't have the ability to resolve a conflict between you and Georgi especially because I'm ignorant of your prior history. My previous policy was that I would simply not look at any of your code and that is what I will go back to.
> 
> 👤 **bartowski1182** replied the **2025-04-13** at **15:47:29**:<br>
> As another outsider without a horse in this race (besides wanting everyone to benefit as much as possible by all the best work), I don't think a simple code comment referencing either the original PR from this repo, or lacking the ability to find one simply, a quick mention of this repo, world detract much if anything from the overall code experience
> 
> In fact, recently when making changes, I've seen code with a comment referencing a PR from other repos, or from llamacpp itself, and these help immensely for tracking down motivations and any potential discussions that went on at the time
> 
> And yes you can git blame, but that becomes cumbersome if there's ever a single refactor
> 
> My unrequested and uneducated 2c

---

👤 **ikawrakow** replied the **2025-04-07** at **11:07:50**:<br>

> My previous policy was that I would simply not look at any of your code and that is what I will go back to.

Yes, of course, as predicted.

---

👤 **jano403** replied the **2025-04-07** at **11:16:19**:<br>

A based thing to do would be to license your repository under AGPL3.0, solves all problems.

> 👤 **ikawrakow** replied the **2025-04-07** at **11:23:15**:<br>
> > A based thing to do would be to license your repository under AGPL3.0, solves all problems.
> 
> Yes, I agree, it would have been better. But I didn't feel like juggling two different licenses, so just went with the original MIT license.
> 
> On the other hand, the final outcome would not have been any different. Mainline will independently discover and implement the improvement I have made here without looking at my changes, not even once. I think this was made very clear by @JohannesGaessler's last comment.
> 
> 👤 **jano403** replied the **2025-04-07** at **11:29:07**:<br>
> Never too late to change it if You ever feel like it.
> Btw, appreciate all the hard work You're doing for quants and speed improvements!
> 
> 👤 **ikawrakow** replied the **2025-04-07** at **11:40:33**:<br>
> I would need to read up on what is the correct way of mixing MIT licensed code with (A)GPL licensed code. Or can you point me to a simple to follow set of instructions?
> 
> 👤 **CISC** replied the **2025-04-07** at **12:00:19**:<br>
> I'm not sure what "problems" that is supposed to fix though? Was the license really the problem?
> 
> 👤 **ikawrakow** replied the **2025-04-07** at **12:06:07**:<br>
> It would have avoided ggerganov talking about the Berne Convention and implying that no copyright notices are required, or putting contributors such as yourself into the difficult position of having to choose between doing the right thing or following his rules.
> 
> 👤 **CISC** replied the **2025-04-07** at **12:15:28**:<br>
> It would have avoided me even considering upstreaming, that's all, the rest is unrelated fallout.
> 
> 👤 **jano403** replied the **2025-04-07** at **12:34:09**:<br>
> > I would need to read up on what is the correct way of mixing MIT licensed code with (A)GPL licensed code. Or can you point me to a simple to follow set of instructions?
> 
> I believe the MIT license is compatible with GPL/AGPL, take a look at https://github.com/LostRuins/koboldcpp for example. The original code would still be MIT licensed but the project as a whole, including Your modifications would be GPL/AGPL licensed.
> ![image](https://github.com/user-attachments/assets/58b0011f-6f53-4cfe-a57f-89101946b1b7)
> 
> 👤 **jano403** replied the **2025-04-07** at **12:35:47**:<br>
> https://www.gnu.org/licenses/license-list.en.html#GPLCompatibleLicenses
> ![image](https://github.com/user-attachments/assets/8d7b887c-fd6d-48e6-a5b8-325110cf1ef5)
> ![image](https://github.com/user-attachments/assets/6ebd73b4-e7f6-4dbe-a75b-d29dc2d05d68)
> 
> edit: As for copyright notices, You could simply add 
> ```
> // Modifications made after <DATE> licensed under GPLv3/AGPLv3
> // AGPL/GPL license
>  // SPDX-License-Identifier: AGPL/GPL
>  //
> ```
> or similar when You make new changes.
> 
> 👤 **ikawrakow** replied the **2025-04-07** at **12:48:51**:<br>
> > It would have avoided me even considering upstreaming, that's all, the rest is unrelated fallout.
> 
> Well, also that. Which have resulted in you having a much less interesting weekend 😄

---

👤 **ikawrakow** replied the **2025-04-07** at **11:24:52**:<br>

@CISC 

I'm sorry you ended up in the middle of this. I hope this has not damaged your relation with, and your ability to contribute to, the `ggml` and `llama.cpp` projects.

> 👤 **CISC** replied the **2025-04-07** at **11:58:00**:<br>
> > I'm sorry you ended up in the middle of this. I hope this has not damaged your relation with, and your ability to contribute to, the `ggml` and `llama.cpp` projects.
> 
> Let's just say this weekend was more interesting than I would have liked. :(