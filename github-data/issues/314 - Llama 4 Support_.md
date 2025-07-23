### 📝 [#314](https://github.com/ikawrakow/ik_llama.cpp/issues/314) - Llama 4 Support?

| **Author** | `Downtown-Case` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-05 |
| **Updated** | 2025-04-10 |

---

#### Description

https://huggingface.co/collections/meta-llama/llama-4-67f0c30d9fe03840bc9d0164

Still waiting for access to the config file, and trying to find the paper... But I wonder if it can use an offloading mechanism similar to deepseek?

It's 10M context, so there must be some architectural difference from Llama 3.3

---

#### 💬 Conversation

👤 **saood06** commented the **2025-04-06** at **00:05:11**:<br>

>It's 10M context, so there must be some architectural difference from Llama 3.3

"A key innovation in the Llama 4 architecture is the use of interleaved attention layers [without positional embeddings](https://arxiv.org/abs/2305.19466). Additionally, we employ [inference time temperature scaling](https://arxiv.org/pdf/2501.19399) of attention to enhance length generalization. We call this the iRoPE architecture, where “i” stands for “interleaved” attention layers, highlighting the long-term goal of supporting “infinite” context length, and “RoPE” refers to the [rotary position embeddings](https://arxiv.org/abs/2104.09864) employed in most layers." from [here](https://ai.meta.com/blog/llama-4-multimodal-intelligence/?utm_source=twitter&utm_medium=organic_social&utm_content=image&utm_campaign=llama4)

This shares a bit from Command-A:

"The model features three layers with sliding window attention (window size 4096) and RoPE for efficient local context modeling and relative positional encoding. A fourth layer uses global attention without positional embeddings, enabling unrestricted token interactions across the entire sequence. " [here](https://huggingface.co/CohereForAI/c4ai-command-a-03-2025)

---

👤 **Downtown-Case** commented the **2025-04-06** at **02:15:26**:<br>

No MLA, which was my faint hope.

Some layers are dense though, so maybe this is a good offloading candidate.

---

👤 **Downtown-Case** commented the **2025-04-06** at **02:15:26**:<br>

No MLA, which was my faint hope.

---

👤 **saood06** commented the **2025-04-06** at **04:45:20**:<br>

> No MLA, which was my faint hope.

"Scout supports upto 10M context. On 8xH100, in bf16 you can get upto 1.4M tokens." from [here](https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/build_with_llama_4.ipynb)

It would be interesting to see how much context the providers end up offering since supporting 10 million seems really difficult.

---

👤 **ikawrakow** commented the **2025-04-08** at **08:04:36**:<br>

I'll look into this in the next days. I did try downloading the Scout variant this morning using `huggingface-cli`, but it errored out. I'll try again later.

---

👤 **Downtown-Case** commented the **2025-04-08** at **16:20:59**:<br>

@ikawrakow I have great success with this:

https://github.com/bodaay/HuggingFaceModelDownloader

It hash checks every file, and will retry each one if it fails or times out.

---

👤 **Downtown-Case** commented the **2025-04-08** at **16:23:04**:<br>

Oh, and Llama 4 seems to be quite bad at longer context, at least in my quick API tests.

---

👤 **ikawrakow** commented the **2025-04-08** at **16:25:48**:<br>

Bad as not producing good answers, or bad as being slow?

---

👤 **saood06** commented the **2025-04-08** at **17:06:37**:<br>

> Oh, and Llama 4 seems to be quite bad at longer context, at least in my quick API tests.

Is it good at short contexts?

---

👤 **Downtown-Case** commented the **2025-04-09** at **14:37:43**:<br>

> Bad as not producing good answers, or bad as being slow?

Bad at producing good answers.

My long context tests are questions about long sets of papers or long stories (like novels) that require the LLM to "grasp" the whole context instead of plucking something out like needle-in-a-haystack tests. For example, "judge these papers against each other," or "describe this character's arc to me," and its... not good. Even at like 70K, much less 1M context.

For reference, Deepseek (even the 32B distills) are quite good at this. Phi is horrendous, Mistral is bad, llama 70B is *OK*, QwQ struggles past 32K once the rope scaling kicks in, and Google Gemini (not Gemma 3, not sure about that) is definitely SOTA.

> Is it good at short contexts?

No idea, lol. Again I was testing over API, not llama.cpp.

---

👤 **Downtown-Case** commented the **2025-04-09** at **14:37:43**:<br>

> Bad as not producing good answers, or bad as being slow?

Bad at producing good answers.

My long context tests are questions about long sets of papers or long stories (like novels) that need it to "understand" lots of whole context instead of pluck something out, like "judge these papers against each other," or "describe this character's arc to me," and its... not good. Even at like 70K, much less 1M context.

---

👤 **saood06** commented the **2025-04-10** at **03:35:44**:<br>

> No idea, lol. Again I was testing over API, not llama.cpp.

I saw this which is a bit suggestive that API quality for this model might have some issues.

![Image](https://github.com/user-attachments/assets/ea6dcee6-9686-46fc-a489-eac6845ff2df)