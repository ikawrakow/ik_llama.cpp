### 🔀 [#176](https://github.com/ikawrakow/ik_llama.cpp/pull/176) - Deepseek V3 support added

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-01-23 |
| **Updated** | 2025-01-23 |

---

#### Description

Very direct port of https://github.com/ggerganov/llama.cpp/pull/11049.

Tested working with IQ4_K_R4 and IQ4_K. No tests so far on any quant that is supported by llama.cpp so that performance can be compared.

Tested on dual socket Xeon E5-2690 v3
Prompt processing:11.5 t/s for IQ4_K, 9.8 t/s IQ4_K_R4
Token generation: 2.75 t/s for IQ4_K, 3.10 t/s for IQ4_K_R4

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [X] Low
  - [ ] Medium
  - [ ] High

---

#### 💬 Conversation

👤 **ikawrakow** submitted a review the **2025-01-23** at **16:09:41**: ✅ `APPROVED`

---

👤 **ikawrakow** commented the **2025-01-23** at **17:00:50**:<br>

@saood06 

Quick question: current `llama.cpp` has this check for Deepseek-V3:
```c++
    } else if (tmpl_contains(LU8("<｜Assistant｜>")) && tmpl_contains(LU8("<｜User｜>")) && tmpl_contains(LU8("<｜end▁of▁sentence｜>"))) {
        return LLM_CHAT_TEMPLATE_DEEPSEEK_3;
```
while the check you added with this PR is
```c++
    else if (tmpl == "deepseek3" || tmpl_contains(LU8("'<｜Assistant｜>' + message['content'] + '<｜end▁of▁sentence｜>'"))) {
```
The check for `tmpl == "deepseek3"` is done before in `llama.cpp`, so this is not an issue, but the remainder is not the same. Is this a problem? Or would it be a problem if I just made it the same as `llama.cpp` ?

---

👤 **saood06** commented the **2025-01-23** at **18:00:03**:<br>

The change you are referencing happened in https://github.com/ggerganov/llama.cpp/commit/ec7f3ac9ab33e46b136eb5ab6a76c4d81f57c7f1 I was not aware of that till now.


>Is this a problem? Or would it be a problem if I just made it the same as llama.cpp ?

 You can change it if you want but both work, based on the chat_templates for the models that have been released.