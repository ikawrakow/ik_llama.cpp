### 🔀 [#628](https://github.com/ikawrakow/ik_llama.cpp/pull/628) - [Draft] Function calling support for Kimi-K2

| **Author** | `iSevenDays` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-07-18 |
| **Updated** | 2025-07-19 |

---

#### Description

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [ ] Low
  - [x] Medium
  - [ ] High
---
The implementation adds support for tool calls.

The reason why I think the feature is important is that it allows users of ik_llama.cpp to use this backend with apps like Claude Code that requires tool calls.

By using simple proxy like this one https://github.com/1rgs/claude-code-proxy (I just found it in github), I could connect Claude Code to ik_llama.cpp using [Kimi-K2 Q2](https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF/tree/main/IQ2_KL)  LLM provided by ubergarm.
In claude-code-proxy you just have to change .env `OPENAI_API_BASE="http://192.168.0.24:8080/v1"`

<img width="570" height="485" alt="image" src="https://github.com/user-attachments/assets/418bdd72-645e-4330-b7d4-52b969157dfe" />

Kimi-k2 uses multiple formats, when not instructed to use specific tool call format.
The list of formats that I observed is in examples/server/function_calls.md file.

<img width="720" height="602" alt="image" src="https://github.com/user-attachments/assets/f093ef6e-4db6-4da9-84f6-a29f5a20b9a5" />

---

#### 💬 Conversation

👤 **ikawrakow** submitted a review the **2025-07-18** at **09:56:32**: ✅ `APPROVED`<br>

Thank you for this! People have been asking for function calling support, but that is not something I'm very familiar with.

LGTM, but I would appreciate at least one other person testing.

I see your location is Leipzig. Have fond memories of this place, having spent 11 years there studying physics, doing a PhD, and staying for my first postdoc position.

---

👤 **iSevenDays** commented the **2025-07-18** at **10:43:28**:<br>

> LGTM, but I would appreciate at least one other person testing.

Thanks! I've done the basic tests, but the model loads too slow from my hdd, so I will test different use cases over the weekend.
I could make it work for the first request, but it seems that multiple requests don't work currently or Kimi-K2 requires a different prompting. I'll debug this more over the weekend and update the PR.

> I see your location is Leipzig. Have fond memories of this place, having spent 11 years there studying physics, doing a PhD, and staying for my first postdoc position.

I live in a beautiful city, thanks! I've been living here for 3 years and have absolutely no regrets!

---

👤 **sousekd** commented the **2025-07-18** at **23:10:28**:<br>

@iSevenDays This seems relevant:

> We've just fixed 2 bugs in Kimi-K2-Instruct huggingface repo. Please update the following files to apply the fix:

- tokenizer_config.json: update chat-template so that it works for multi-turn tool calls.
- tokenization_kimi.py: update encode method to enable encoding special tokens.

https://x.com/Kimi_Moonshot/status/1945050874067476962

---

👤 **mtcl** commented the **2025-07-19** at **16:30:45**:<br>

This is very exciting! I would much rather use a native function calling!

---

👤 **iSevenDays** commented the **2025-07-19** at **17:10:18**:<br>

I took a look at how llama.cpp implements tool calling support and the task is much more complicated that I thought. Especially, the streaming part.
I'll keep you updated.

---

👤 **mtcl** commented the **2025-07-19** at **17:42:16**:<br>

> I took a look at how llama.cpp implements tool calling support and the task is much more complicated than I thought. Especially, the streaming part.
> I'll keep you updated.

That would be really amazing! ik_llama + tool calling will be a dream come true for me!