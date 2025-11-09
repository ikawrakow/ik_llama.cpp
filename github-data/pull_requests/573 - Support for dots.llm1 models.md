## 🔀 [Pull Request #573](https://github.com/ikawrakow/ik_llama.cpp/pull/573) - Support for dots.llm1 models

| **Author** | `saood06` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `s6/dots` |
| **Target Branch** | `main` |
| **Created** | 2025-07-03 |
| **Updated** | 2025-07-10 |
| **Merged** | 2025-07-10 |

---

## 📄 Description

Port of https://github.com/ggml-org/llama.cpp/pull/14118

It compiles. Testers welcome.

Edit: Tested myself a tiny bit (more testers still welcome), see comment below.

Huggingface link to models: [instruct](https://huggingface.co/rednote-hilab/dots.llm1.inst), [base](https://huggingface.co/rednote-hilab/dots.llm1.base)

---

## 💬 Conversation

👤 **saood06** commented on **2025-07-03** at **03:44:01**

Tested a bit in the cli, seems to work.

Command:
`./bin/llama-cli -m /mnt/sda/dots_inst/Dots_Inst-128x8.7B-BF16.gguf -s 12345 -p "The meaning of life" -t 48 --numa distribute -n 32 -c 8192 -fa`

Output: 

The meaning of life is to find your gift. The purpose of life is to give it away.  — Willam James

This is as much as I had patience for, warmup seems to not actually load in all the experts and so tokens trickle in very slowly, not sure if that is the norm for CLI on MoE models (I know this isn't an issue for me with Deepseek models on server or sweep-bench).

I also noticed it is wrongly labeled as it says `model ftype = IQ1_S - 1.5625 bpw` even though it is a `BF16`, and found the issue. When I updated constants.py for LlamaFileType I used ggml.h instead of llama.h (only now realized that both have `ftype` info and they differ [not sure why?])

---

👤 **firecoperana** commented on **2025-07-03** at **04:52:49**

I am testing using UD-Q4_K_XL, and it is working. I notice an issue that if I leave system prompt empty, sometimes the response becomes unrelated to my question. With system prompt, it is fine.  Do you also see this? I have the same issue when I run it from mainline.

---

👤 **saood06** commented on **2025-07-03** at **04:59:14**

> I am testing using UD-Q4_K_XL, and it is working.

Thanks.

>I notice an issue that if I leave system prompt empty, sometimes the response becomes unrelated to my question. With system prompt, it is fine. Do you also see this? I have the same issue when I run it from mainline.

If it exists in mainline then maybe it is a problem with the model? I haven't seen it but I haven't tested the model further than my comment above.

---

👤 **ikawrakow** started a conversation on `src/llama.cpp` on **2025-07-03** at **06:18:24**

I think you need to remove this line. We are not reshaping `V` as mainline because our attention implementation is different from theirs (and theirs was like ours until 2 or 3 months ago).

> 👤 **saood06** replied on **2025-07-09** at **17:29:30**
> 
> Commented it out (and the then redundant `cb`), and tested and it is working.

---

👤 **ikawrakow** requested changes on this pull request 🔄 on **2025-07-03** at **06:19:04**

---

👤 **firecoperana** commented on **2025-07-03** at **15:20:55**

I also see that the response will pause for a few seconds whenever it generates a comma, which will more than half the generation speed. If I prompt it to avoid outputting comma in the response, I don't see any pause in response. Mainline does not have this issue because it does not output comma in the response. 

Screenshot of the quant that I use:
![image](https://github.com/user-attachments/assets/625a1221-be6d-4e1e-8924-d11822d696c6)

BOS token is ",", which should be changed to -1 according to this post:
https://huggingface.co/gghfez/dots.llm1.inst-GGUF/discussions/1

---

👤 **saood06** commented on **2025-07-03** at **22:55:12**

> I also see that the response will pause for a few seconds whenever it generates a comma, which will more than half the generation speed. If I prompt it to avoid outputting comma in the response, I don't see any pause in response. Mainline does not have this issue because it does not output comma in the response.

Interesting, you are using `Q4_K_XL`. There is a lot of reporting about issues with certain quants of some Qwen based models (and this is one of those) pausing whenever they encounter a comma.

2 users here who narrow it down to certain quants of some Qwen based models:
https://github.com/ikawrakow/ik_llama.cpp/issues/464#issuecomment-2925026167
https://github.com/ikawrakow/ik_llama.cpp/issues/464#issuecomment-2927631215

2 users here who identify it happening with commas, and causing performance issues:
https://github.com/ikawrakow/ik_llama.cpp/issues/476#issuecomment-2933070214
https://github.com/ikawrakow/ik_llama.cpp/issues/476#issuecomment-2972846150 (this one even shows the effect on video)

The first sighting on the github I know about:
https://github.com/ikawrakow/ik_llama.cpp/issues/380#issuecomment-2850596618

I'm not sure what the root cause is, but I wouldn't investigate it with this model, I think the smallest model it is reported on is `Qwen3-30B-A3B-128K-UD-Q4_K_XL`.

---

👤 **firecoperana** commented on **2025-07-03** at **23:30:53**

The following fix works for me:
![image](https://github.com/user-attachments/assets/71ee67bb-f1e6-4de5-bd2a-a3ce4d44f897)
Not sure if there is better way.

---

👤 **saood06** commented on **2025-07-04** at **00:05:25**

>  Not sure if there is better way.

That fix is only for the incorrect BOS token (not the comma's causing pausing, right?), which to me seems like an issue with existing models caused by the convert script which is where the fix should happen (with workarounds like [this](https://huggingface.co/gghfez/dots.llm1.inst-GGUF/discussions/1 for existing models) .

Both the `config.json` and `tokenizer_config.json` are set to null, which makes it take the default, but that doesn't seem to be correct for this model at least.

---

👤 **firecoperana** commented on **2025-07-04** at **00:10:41**

Without the fix, the model uses comma as BOS token that causes the pause, as least for the quant I'm using. See the screenshot I posted. Id 11 is the comma. After I set to null, comma is not used as BOS token.

---

👤 **saood06** commented on **2025-07-04** at **00:24:53**

> Without the fix, the model uses comma as BOS token that causes the pause, as least for the quant I'm using. See the screenshot I posted. Id 11 is the comma. After I set to null, comma is not used as BOS token.

Well the comma still causes a pause (I'm assuming) even if you avoid encountering it from the BOS token by setting the BOS token.

I've seen the screenshot you posted, and I also see the wrong BOS token (`BOS token = 11 ','`) in my own GGUF that I converted as part of the testing here (from safetensors to BF16 GGUF).

Using `--override-kv tokenizer.ggml.bos_token_id=int:-1` like you linked above fixes it for affected models, but for future models to not be affected I think the convert script needs to explicitly set it, without changing the default like the `llama.cpp` change you suggested does.

---

👤 **ikawrakow** commented on **2025-07-09** at **08:31:56**

@saood06 What are your plans with this PR? You are disagreeing with the `V` reshaping comment, or is it about the `BOS` token, or perhaps both?

---

👤 **saood06** commented on **2025-07-09** at **17:45:47**

> @saood06 What are your plans with this PR?

Sorry kept pushing off testing this more, but I just pushed a commit with both the recommended changes.

I tested all four `-fa` and `-fmoe` combinations and it works, (without the V cur changes, non FA was outputting garbage).

>You are disagreeing [...] about the `BOS` token

I still think the better solution would have been for the convert script to set it to `-1` when config.json has it set to `NULL` instead of leaving it to be set to default and changing the default for this architecture, but given the fact that every GGUF I saw on huggingface has this issue, changing the default so that users don't have to set `--override-kv tokenizer.ggml.bos_token_id=int:-1` (assuming they know to do that) or some other workaround to use existing GGUFs makes sense.

I also changed the warmup behavior to work with this model (a MoE without a BOS token), it is still the same hacky solution but now it does account for models without a BOS token, and it did warmup properly for me now (not sure why it wasn't with BOS set to [token id 11/`,`]).

Edit: Also handled the merge conflicts.

---

👤 **ikawrakow** approved this pull request ✅ on **2025-07-10** at **06:31:53**