### 🗣️ [#395](https://github.com/ikawrakow/ik_llama.cpp/discussions/395) - Why does imatrix not tokenize special tokens?

| **Author** | `bartowski1182` |
| :--- | :--- |
| **Created** | 2025-05-07 |
| **Updated** | 2025-05-09 |

---

#### Description

Recently there's been some discussion (and I've also experimented slightly) around adding chat tokens to the imatrix dataset and tokenizing them, a change from the default behaviour, so I was curious why the original implementation avoided tokenizing them

Was it just an arbitrary decision or was there a reason at the time?

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2025-05-08** at **05:21:04**:<br>

When the `imatrix` tool was written handling of chat, special tokens, etc., was extremely immature/non-existent in `llama.cpp` . If you look at the `llama_tokenize` function in `common` that is being used by the `imatrix` tool to tokenize the calibration data, you will see that the `parse_special` argument was added well after the `imatrix` tool was merged. It was added with a default value of `false`, so that defined the `imatrix` tool behavior with special tokens as this argument is missing in the `imatrix` call to `::lama_tokenize`. By the time `llama_tokenize` got the ability to parse special tokens I had left the `llama.cpp` project, so somebody else needed to notice, investigate, and possibly change.

Back then I had the concept that the calibration data for chat/instruction tuned models need to contain actual instruction tuning datasets. And, instead of blindly dividing the calibration data into chunks of `n_ctx` tokens, the chunks needed to be individual request-response pieces (or series of related request-response chunks in a conversation). But then everybody became an expert on `imatrix` calibration data, people started using the `imatrix` tool the way it is for chat models and it seemed to work OK, so I never followed up.

In any case, it would be interesting to see if including special tokens, using non equal-size chunks, etc., in the `imatrix` calibration data would improve the quality of quantized models.

---

👤 **ikawrakow** replied the **2025-05-09** at **08:46:05**:<br>

@bartowski1182 I see you submitted [this PR](https://github.com/ggml-org/llama.cpp/pull/13389) in mainline.

You are welcome.

> 👤 **bartowski1182** replied the **2025-05-09** at **12:33:00**:<br>
> Ah did I not send that reply here first? Sorry, I had one typed up
> 
> That makes perfect sense though! Do you think you'd want the same thing here? Was planning to open one up in each assuming it made sense, it seems like a nice idea for A/B testing anyways, but figured I'd double check with the original architect that there wasn't something glaringly obvious I was missing
> 
> Thanks again for the input!
> 
> 👤 **bartowski1182** replied the **2025-05-09** at **12:42:35**:<br>
> Truly did not mean to just grab knowledge and run, that's a terrible look, hence I meant to ask if I could contribute the same here so that it wouldn't just be a one-sided deal (not that it's a complex change from me, but just the principle of it, it's not in good taste to open a discussion, get your insight, and run to mainline without saying anything, that isn't my style but it's exactly what I did in this case)
> 
> 👤 **ikawrakow** replied the **2025-05-09** at **12:42:53**:<br>
> > Do you think you'd want the same thing here?
>  
> Most people are using mainline `llama.cpp` to compute imatrix data, so it is not critical to have this here.
> 
> I'm waiting to see if the mainline developers will independently discover what's wrong with the imatrix calculation after their change to support MLA. After they have independently discovered it, or when enough time has passed, I'll make the change here, and at that point I can also put in the ability to use special tokens. Do you hear complains from users about reduced model quality after the MLA change?
> 
> 👤 **bartowski1182** replied the **2025-05-09** at **12:47:29**:<br>
> > Do you hear complains from users about reduced model quality after the MLA change
> 
> No I didn't hear anything about that yet, but MLA has its own can of worms with speed so I had personally been avoiding remaking those models that have MLA since, hoping for a resolution...
> 
> Now I almost want to go on a hunt for it, but know it's gonna go right over my head as with other imatrix code :')
> 
> Without looking directly at your commit history I doubt anyone in mainline will figure it out, but who knows
> 
> I do know that I like your algorithm for some semi incomplete experts, seems reasonable to have some wiggle room there, especially if after 200k tokens of imatrix it's still not being activated quite enough
> 
> 👤 **ikawrakow** replied the **2025-05-09** at **12:48:22**:<br>
> > Truly did not mean to just grab knowledge and run, that's a terrible look, hence I meant to ask if I could contribute the same here so that it wouldn't just be a one-sided deal (not that it's a complex change from me, but just the principle of it, it's not in good taste to open a discussion, get your insight, and run to mainline without saying anything, that isn't my style but it's exactly what I did in this case)
> 
> No worries. I know you are not free to mention my name in the mainline repository, else your PR will have the same fate as [that one](https://github.com/ggml-org/llama.cpp/pull/12727)
> 
> 👤 **bartowski1182** replied the **2025-05-09** at **12:55:14**:<br>
> > else your PR will have the same fate as that one
> 
> I'd *like* to think that's not the reason, but rather the annoying complexity level of that function in general and excitement for a new feature (though the feature does miss out on an important part, counting discrete layers ahead of time and applying variable quantization automatically..)
> 
> But who knows, it's not my drama to unpack, so much as I wish we could all get along in a nice Kumbaya circle and contribute to the open world together, I know I'm naive ;)
> 
> 👤 **ikawrakow** replied the **2025-05-09** at **13:03:17**:<br>
> It has never been the style of the `llama.cpp` project to wait for the perfect solution before merging a useful change.
> 
> Your PR is immensely helpful to anyone using mainline `llama.cpp` and making their own quantized MoE models.  
> 
> Sadly, there is only one possible conclusion from these two observations.