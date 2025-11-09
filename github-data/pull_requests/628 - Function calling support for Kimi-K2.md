## 🔀 [Pull Request #628](https://github.com/ikawrakow/ik_llama.cpp/pull/628) - Function calling support for Kimi-K2

| **Author** | `iSevenDays` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `function_calling` |
| **Target Branch** | `main` |
| **Created** | 2025-07-18 |
| **Updated** | 2025-07-26 |
| **Merged** | 2025-07-23 |

---

## 📄 Description

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [ ] Low
  - [ ] Medium
  - [x] High
---
The implementation adds support for tool calls.

The reason why I think the feature is important is that it allows users of ik_llama.cpp to use this backend with apps like Claude Code that requires tool calls.

By using simple proxy like this one https://github.com/1rgs/claude-code-proxy (I just found it in github), I could connect Claude Code to ik_llama.cpp using [Kimi-K2 Q2](https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF/tree/main/IQ2_KL)  LLM provided by ubergarm.
In claude-code-proxy you just have to change .env `OPENAI_API_BASE="http://192.168.0.24:8080/v1"`

<img width="570" height="485" alt="image" src="https://github.com/user-attachments/assets/418bdd72-645e-4330-b7d4-52b969157dfe" />

I had to port llama.cpp function tool calls support. The most difficult part was to port streaming and json healing.

<img width="720" height="602" alt="image" src="https://github.com/user-attachments/assets/f093ef6e-4db6-4da9-84f6-a29f5a20b9a5" />

---

## 💬 Conversation

👤 **ikawrakow** approved this pull request ✅ on **2025-07-18** at **09:56:32**

Thank you for this! People have been asking for function calling support, but that is not something I'm very familiar with.

LGTM, but I would appreciate at least one other person testing.

I see your location is Leipzig. Have fond memories of this place, having spent 11 years there studying physics, doing a PhD, and staying for my first postdoc position.

---

👤 **iSevenDays** commented on **2025-07-18** at **10:43:28**

> LGTM, but I would appreciate at least one other person testing.

Thanks! I've done the basic tests, but the model loads too slow from my hdd, so I will test different use cases over the weekend.
I could make it work for the first request, but it seems that multiple requests don't work currently or Kimi-K2 requires a different prompting. I'll debug this more over the weekend and update the PR.

> I see your location is Leipzig. Have fond memories of this place, having spent 11 years there studying physics, doing a PhD, and staying for my first postdoc position.

I live in a beautiful city, thanks! I've been living here for 3 years and have absolutely no regrets!

---

👤 **ubergarm** commented on **2025-07-18** at **16:38:14**

> I could make it work for the first request, but it seems that multiple requests don't work currently or Kimi-K2 requires a different prompting. I'll debug this more over the weekend and update the PR.> 

Oh hej this is exciting! I believe we have a PR open for this https://github.com/ikawrakow/ik_llama.cpp/issues/407#issuecomment-2889059989 where some folks were trying to use a reverse proxy / wrapper to handle it similar to claude-code-proxy perhaps.

I don't use tool calling myself, but did notice when adding Kimi-K2-Instruct PR that I left out one section for the chat endpoint for the `"role": "tool"`: https://github.com/ggml-org/llama.cpp/pull/14654#issuecomment-3074893927

So if it expects llama-server to handle the template internally that `"role": "tool"` might not be applied. But if you're using the text completions endpoint and doing your own template it might not matter.

---

👤 **sousekd** commented on **2025-07-18** at **23:10:28**

@iSevenDays This seems relevant:

> We've just fixed 2 bugs in Kimi-K2-Instruct huggingface repo. Please update the following files to apply the fix:
>
>- tokenizer_config.json: update chat-template so that it works for multi-turn tool calls.
>- tokenization_kimi.py: update encode method to enable encoding special tokens.

https://x.com/Kimi_Moonshot/status/1945050874067476962

---

👤 **mtcl** commented on **2025-07-19** at **16:30:45**

This is very exciting! I would much rather use a native function calling!

---

👤 **iSevenDays** commented on **2025-07-19** at **17:10:18**

I took a look at how llama.cpp implements tool calling support and the task is much more complicated than I thought. Especially, the streaming part.
I'll keep you updated.

---

👤 **mtcl** commented on **2025-07-19** at **17:42:16**

> I took a look at how llama.cpp implements tool calling support and the task is much more complicated than I thought. Especially, the streaming part.
> I'll keep you updated.

That would be really amazing! ik_llama + tool calling will be a dream come true for me!

---

👤 **iSevenDays** commented on **2025-07-22** at **16:16:11**

I had to port llama.cpp function tool calls support.

Here is branch of Claude Proxy that you can use with ik_llama.cpp and Claude code.

Steps to test this PR
1. Clone https://github.com/iSevenDays/claude-code-proxy
2. Run the proxy
```
uv run uvicorn server:app --host 0.0.0.0 --port 8082
```
3. Open .env inside claude proxy
```
OPENAI_API_BASE="http://192.168.0.24:8080/v1"
PREFERRED_PROVIDER="openai"
BIG_MODEL="Kimi-K2"
SMALL_MODEL="Kimi-K2"
```
4. The model name is important, so set it to kimi-k2 to enable tool parsing from ik_llama.cpp
5. Test with Claude Code
```
ANTHROPIC_BASE_URL=http://localhost:8082 claude "list files"
```

I'm doing more tests in the meantime.

---

👤 **iSevenDays** commented on **2025-07-23** at **09:00:50**

I added Qwen3 tool calling support.
From my tests, Kimi-K2 uses tools better and Qwen3 fails to use tools for Claude Code.

---

👤 **iSevenDays** commented on **2025-07-23** at **09:06:45**

@ikawrakow I have backported tool calling support. I'm not sure if I can make PR smaller, because the feature in llama.cpp is quite complicated. 
I'd be glad if somebody can also do real world tests.

I suggest using Kimi-K2 model with Claude Code using these steps https://github.com/ikawrakow/ik_llama.cpp/pull/628#issuecomment-3103627677 

It seems to work fine, at least it can call tools when I explicitly ask for it.

---

👤 **ikawrakow** commented on **2025-07-23** at **09:13:50**

I think there was a lot of interest for this, so hopefully we will have a few people testing the PR. Hopefully today, so I can merge before going on vacation tomorrow.

---

👤 **iSevenDays** commented on **2025-07-23** at **09:17:20**

@ikawrakow I'll be happy to work on your requests for this PR to get it merged.
I followed the strategy of porting llama.cpp as close as possible.

---

👤 **xldistance** commented on **2025-07-23** at **09:27:45**

Looking forward to qwen3's tool call

---

👤 **iSevenDays** commented on **2025-07-23** at **10:10:58**

I have added DeepSeek-R1 tool calling support.
The following LLM works just fine. It takes often 2 iterations to do the tool call, but Claude Code handles that automatically.

```
numactl --interleave=all ./build/bin/llama-server \
                         --alias DeepSeek-R1T2 \
                         --model /root/models/DeepSeek-TNG-R1T2-Chimera-GGUF/IQ3_KS/IQ3_KS/DeepSeek-TNG-R1T2-Chimera-IQ3_KS-00001-of-00007.gguf \
                         -rtr \
                         --ctx-size 102400 \
                         -ctk q8_0 \
                         -mla 3 -fa \
                         -amb 512 \
                         -fmoe \
                         --temp 0.6 \
                         --top_p 0.95 \
                         --n-gpu-layers 63 \
                         --override-tensor "blk\.([0-5])\.ffn_.*=CUDA0,exps=CPU" \
                         --parallel 1 \
                         --threads 16 \
                         --host 0.0.0.0 \
                         --port 8080 \
                         --min_p 0.01 \
                         --numa distribute \
                         --threads-batch 32 \
                         --no-mmap \
                         -b 8192 -ub 8192
```

---

👤 **xldistance** commented on **2025-07-23** at **10:43:12**

@iSevenDays json-partial.h
json-partial.cpp
regex-partial.h
regex-partial.cpp Missing documents

---

👤 **iSevenDays** commented on **2025-07-23** at **11:12:28**

@xldistance thanks for the feedback, the files are there and can be compiled successfully.

For those who are testing with Claude Code, here are my suggestions:
Kimi-K2 works the best, and is very fast, uses tools.
DeepSeek-TNG-R1T2-Chimera works, but too often it times out on my Dell R740 48GB 4090D.
Qwen3-235B-A22B-Instruct-2507-GGUF (pure-IQ4_KS from ubergarm) doesn't want to use tools

---

👤 **xldistance** commented on **2025-07-23** at **11:14:21**

@iSevenDays I use qwen3-coder-480b on top of ccr code

---

👤 **iSevenDays** commented on **2025-07-23** at **11:18:50**

@xldistance just make sure to set the correct name of LLM in env and in llama-server.
I enabled name matching e.g. the following triggers additional tool calling in system prompt to let the model know how to use tools properly. I ported the behavior from llama.cpp. Llama.cpp uses more complex system btw.
The following names would work:
Qwen3-235b
DeepSeek-R1
Kimi-K2
Kimi_K2 

I'll check qwen3-coder-480b that was recently uploaded https://huggingface.co/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/tree/main/IQ2_KS

---

👤 **ikawrakow** commented on **2025-07-23** at **16:11:36**

Well, I'll just merge it then.

---

👤 **iSevenDays** commented on **2025-07-24** at **12:14:41**

@xldistance I found one issue with function tool calls when using LLM with Claude Code.
Please check this PR https://github.com/ikawrakow/ik_llama.cpp/pull/643 to have the latest updates. Now I can use Qwen3 with Claude Code Proxy as well.

---

👤 **randoentity** commented on **2025-07-24** at **14:54:27**

FWIW: tested and working with local qwen. Haven't run into the issue above yet. I'm not using the proxy/router from above though. Is there any way to make this work with jinja templates and not having the model name hardcoded?

---

👤 **mtcl** commented on **2025-07-24** at **16:09:14**

> FWIW: tested and working with local qwen. Haven't run into the issue above yet. I'm not using the proxy/router from above though. Is there any way to make this work with jinja templates and not having the model name hardcoded?

What's the exact command that you used to start the server? Can you please share?

---

👤 **randoentity** commented on **2025-07-24** at **21:15:54**

@mtcl There's nothing special to it, look at isevendays' example above, just use `--alias Qwen3-235b` instead (but just qwen should be sufficient). Also check out the documentation added in this PR as it has an example of what the request should look like. Note that the model name is significant.

---

👤 **city96** commented on **2025-07-26** at **12:42:17**

I did an update today and noticed token streaming wasn't working on latest master. I've tracked it down to this PR, with the commit right before it working.

When token streaming is disabled, the reply is generated as usual and appears once generation finishes. When I enable token streaming, the generation still finishes in the background, but I never get any output. I was testing with an old version of sillytavern, but it seems reproducible in [mikupad](https://github.com/lmg-anon/mikupad) which is probably easier to reproduce.

I get the same issue on Kimi, Deepseek V3, and even just random models like gemma:

```
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-server -m /mnt/models/llm/gemma-3-27b-it-q6_k.gguf -c 16384 -ngl 99
```

---

👤 **iSevenDays** commented on **2025-07-26** at **12:45:24**

@city96 could you please check this PR https://github.com/ikawrakow/ik_llama.cpp/pull/652 and could you please provide a minimum reproducible example? At best, using some small LLM. Then I could check and verify it quickly.

I'm currently testing the PR above and I use both streaming and non-streaming mode with Kimi-K2 model and I didn't notice any issues, but I would gladly help you resolve the issue if there was a regression.

---

👤 **city96** commented on **2025-07-26** at **13:14:50**

I tested your linked PR, but still saw the same problem. I think I found the issue, though. It's this change that this PR makes:

<img width="686" height="241" alt="image" src="https://github.com/user-attachments/assets/746b4287-0c2a-45d1-976c-a6ab9df5d204" />

On latest master that line is here. Changing it back fixes streaming.

https://github.com/ikawrakow/ik_llama.cpp/blob/4e9c78c039601c99541726d95216e3aa7bfda742/examples/server/server.cpp#L1621

Not sure what the logic is in mainline llama.cpp for streaming, but I am using text completion instead of the chat completion endpoint. I assume this is likely why it wasn't caught, since most people probably use the openai compatible one.

For a reproducible example, you can start the ik_llama.cpp server example using any model (I used gemma 27B for testing, but any model should work). Connect to it via mikupad and enter a simple query, enable token streaming, then hit "predict" at the bottom. I can try and make a pure python example as well if it helps.

<img width="881" height="270" alt="image" src="https://github.com/user-attachments/assets/be7e5462-a5ca-43e5-8e06-a015ad44761e" />

---

👤 **iSevenDays** commented on **2025-07-26** at **14:45:12**

@city96 could you please test the change in this PR https://github.com/ikawrakow/ik_llama.cpp/pull/654 ?
I think you have correctly identified the issue, but I'll be able to test that change only later today.

---

👤 **city96** commented on **2025-07-26** at **18:31:30**

@iSevenDays I can confirm that the newest PR does indeed fix token streaming on the text completion endpoint for me, thank you.