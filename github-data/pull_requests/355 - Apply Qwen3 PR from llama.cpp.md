## 🔀 [Pull Request #355](https://github.com/ikawrakow/ik_llama.cpp/pull/355) - Apply Qwen3 PR from llama.cpp

| **Author** | `bharrisau` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `qwen3` |
| **Target Branch** | `main` |
| **Created** | 2025-04-29 |
| **Updated** | 2025-04-29 |
| **Merged** | 2025-04-29 |

---

## 📄 Description

I've just ported over the Qwen3 PR. So it is missing the layers/model type, and does not have tests, etc.


- [ ] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [ ] Low
  - [ ] Medium
  - [X] High

---

## 💬 Conversation

👤 **ikawrakow** commented on **2025-04-29** at **06:55:54**

Thanks! I was just in the process of doing the same.

Does `convert_hf_gguf.py` work with this model?

---

👤 **ikawrakow** started a conversation on `gguf-py/gguf/constants.py` on **2025-04-29** at **07:05:07**

You are missing the `QWEN3` and `QWEN3MOE` enum entries further up in `class MODEL_ARCH(IntEnum)`

---

👤 **ikawrakow** approved this pull request ✅ on **2025-04-29** at **07:06:58**

---

👤 **ikawrakow** commented on **2025-04-29** at **08:02:04**

OK, I'll merge this and will add the missing enum entries separately.

---

👤 **bharrisau** commented on **2025-04-29** at **08:28:30**

Ok - my other concern was that `LLM_ARCH_GRANITE = 46` line. Wasn't sure if I could remove that or not, but as I added more enum entries above it, having it hard coded didn't work.

---

👤 **bharrisau** commented on **2025-04-29** at **08:29:34**

I've only tested that the MOE works.

```
# ./build/bin/llama-cli -m ~/models/Qwen3-30B-A3B-Q6_K.gguf --numa distribute -t 16 --prompt "<|im_start|>system\nWho was prime minister of Australia in 2008?<|im_end|>\n<|im_start|>assistant\n" -fa -fmoe -c 16384 --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 -ctk q8_0       

system
Who was prime minister of Australia in 2008?
assistant
<think>
Okay, so I need to find out who the Prime Minister of Australia was in 2008. Let me start by recalling what I know. I remember that Australia has had several Prime Ministers over the years. From what I've heard, the country has had leaders like Bob Hawke, Malcolm Fraser, and more recently, maybe someone like Tony Abbott or Julia Gillard. But I'm not sure about the exact years.

Wait, 2008 is a specific year. Let me think. I think the Prime Minister before Julia Gillard was Kevin Rudd. But I need to check the timeline. Let me try to remember. Kevin Rudd was Prime Minister from 2007 to 2010, right? So if he was in office in 2007, then he would have been the PM in 2008 as well. But I should confirm that.

Alternatively, maybe there was a change in 2008. Let me think about major events. The Global Financial Crisis happened around 2008, so maybe that's when there was a change in leadership. But I think Kevin Rudd was still PM during that time. Then, in 2010, he was replaced by Julia Gillard. So in 2008, the PM would be Kevin Rudd.

Wait, but I should make sure. Maybe I'm mixing up the dates. Let me try to recall the exact years. Kevin Rudd became Prime Minister in 2007, after the 2007 election. He was the leader of the Australian Labor Party. Then, in 2010, he was replaced by Julia Gillard. So between 2007 and 2010, he was PM. Therefore, in 2008, he was still in office.

Another way to check: I remember that the 2008 Summer Olympics were held in Beijing, but that's not directly related. However, the Australian government under Rudd was involved in some policies, like the carbon pricing mechanism, which was introduced later, but maybe that's after 2008.

Alternatively, maybe there was a leadership challenge in 2008. But I think Rudd remained PM until 2010. So the answer should be Kevin Rudd. Let me see if there's any chance of confusion. For example, if there was a caretaker PM or something, but I don't think so. The PM in 2008 would definitely be Kevin Rudd.

I think that's correct. To be thorough, maybe I can think of other names. For example, Malcolm Turnbull was PM later, but that was after 2013. So no. So yes, Kevin Rudd was the Prime Minister of Australia in 2008.
</think>

The Prime Minister of Australia in 2008 was **Kevin Rudd**. He served as the 26th Prime Minister from December 2007 to June 2010. Rudd led the Australian Labor Party (ALP) to victory in the 2007 federal election, ending 11 years of conservative governance under John Howard. His tenure included significant policies such as the introduction of a carbon pricing mechanism and responses to the global financial crisis. He was succeeded by Julia Gillard in 2010.

**Answer:** Kevin Rudd. [end of text]
```

---

👤 **bharrisau** commented on **2025-04-29** at **08:30:44**

And a no-think example

```
# ./build/bin/llama-cli -m ~/models/Qwen3-30B-A3B-Q6_K.gguf --numa distribute -t 16 --prompt "<|im_start|>system\nWho was prime minister of Australia in 2008?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" -fa -fmoe -c 16384  -ctk q8_0 --temp 0.7 --top-k 20 --top-p 0.8 --min-p 0
system
Who was prime minister of Australia in 2008?
assistant
<think>

</think>

In 2008, the Prime Minister of Australia was **Kevin Rudd**.

He took office on **December 3, 2007**, after leading the Australian Labor Party to a victory in the federal election. He served as Prime Minister until **June 24, 2010**, when he was replaced by **Julia Gillard**. [end of text]                                          
llama_print_timings:        load time =    2157.68 ms
llama_print_timings:      sample time =      10.92 ms /    78 runs   (    0.14 ms per token,  7140.90 tokens per second)
llama_print_timings: prompt eval time =     558.01 ms /    25 tokens (   22.32 ms per token,    44.80 tokens per second)                                                                                                                                                      llama_print_timings:        eval time =   10152.31 ms /    77 runs   (  131.85 ms per token,     7.58 tokens per second)
llama_print_timings:       total time =   10866.73 ms /   102 tokens
```

---

👤 **ikawrakow** commented on **2025-04-29** at **09:07:24**

I also tested before merging and it seemed to be working correctly.