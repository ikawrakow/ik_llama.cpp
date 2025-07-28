## 🔀 [Pull Request #639](https://github.com/ikawrakow/ik_llama.cpp/pull/639) - Fix pauses after a comma

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/fix_comma_pauses` |
| **Target Branch** | `main` |
| **Created** | 2025-07-23 |
| **Updated** | 2025-07-23 |
| **Merged** | 2025-07-23 |

---

## 📄 Description

Closes [#464](https://github.com/ikawrakow/ik_llama.cpp/issues/464).

It seems there are models out there where the BOS token id is the same as the token ID of a comma (11 in the case of Qwen3 MoE models). This results in interpreting a comma during token generation as warm up run, which then results in using all experts, which makes the run time for the next token much longer, which then looks like a pause in the generation. The logic to use all experts during warm up was added in [#198](https://github.com/ikawrakow/ik_llama.cpp/issues/198) to improve the user experience with very large MoE models.  

This PR fixes the issue by checking how many tokens have been evaluated in the given context and only creating a warm up graph if this is zero (in addition to the other conditions to detect a warm up run).

---

## 💬 Conversation

👤 **saood06** commented on **2025-07-23** at **09:37:55**

I was just compiling something similar to this (checking from the llama_kv_cache object) on top of adding support for the flag. Your solution is much cleaner.

---

👤 **saood06** approved this pull request ✅ on **2025-07-23** at **09:39:08**

---

👤 **ubergarm** commented on **2025-07-23** at **16:15:15**

@ikawrakow 

Yes this seems to fix the issue. I notice that now with this compiled in the first chat is *much* faster and subsequent chats no longer seem to pause after `,`. 

I'm spreading the word to update and recompile https://huggingface.co/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/discussions/1#6880eb57b50b0bb883e58f44  and no more need for that `--override-kv tokenizer.ggml.bos_token_id=int:151643` business

Thanks!