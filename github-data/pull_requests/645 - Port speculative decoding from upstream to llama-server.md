### [Pull Request #645](https://github.com/ikawrakow/ik_llama.cpp/pull/645) - Port speculative decoding from upstream to llama-server

| **Author** | `g2mt` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-07-25 |
| **Updated** | 2025-07-25 |
| **Assignees** | `saood06` |

---

#### Description

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [ ] Low
  - [x] Medium
  - [ ] High

Related to #322 

This is a port of the speculative decoding function for llama-server from the upstream code base.

Changes:

- Updated llama-server source code
- Added several functions needed for speculative decoding.
- Add prefixes to KV cache tensors to  support loading of multiple models

I used Qwen3-235B in this PR.

---

#### 💬 Conversation

👤 **saood06** commented on **2025-07-25** at **05:15:48**

Thank you for doing this. I can test/review/assist if you need.

---

👤 **saood06** commented on **2025-07-25** at **05:18:58**

Also are you aware this: https://github.com/ikawrakow/ik_llama.cpp/blob/main/examples/speculative/speculative.cpp exists.

---

👤 **g2mt** commented on **2025-07-25** at **05:26:10**

I got the server to compile, but when loading Qwen 2.5 1.5b with the 0.5b version as the draft, I get this error:

```
ggml_backend_alloc_ctx_tensors_from_buft: all tensors in the context are already allocated
llama_kv_cache_init: failed to allocate buffer for kv cache
llama_new_context_with_model: llama_kv_cache_init() failed for self-attention cache
llama_init_from_gpt_params: error: failed to create context with model 'Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf'
 ERR [              load_model] failed to load draft model | tid="140650859190528" timestamp=1753420591 model="Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf"
```

GDB says it occurred in this llama_init_from_gpt_params call:

```cpp
            llama_init_result llama_init_dft = llama_init_from_gpt_params(params_dft);
```

I wonder if llama_kv_cache_init is unable to load tensors with the same name. I'll try and fix the code later.

---

👤 **g2mt** commented on **2025-07-25** at **05:27:44**

> Also are you aware this: https://github.com/ikawrakow/ik_llama.cpp/blob/main/examples/speculative/speculative.cpp exists.

I am aware of the example. I'll check it later.

---

👤 **saood06** commented on **2025-07-25** at **05:34:38**

>I am aware of the example. I'll check it later.

Sorry. I forgot my history. The common one (introduced here: https://github.com/ggml-org/llama.cpp/pull/10362) was done before server: https://github.com/ggml-org/llama.cpp/pull/10455. The common implementation was made to be simpler to understand and work with which is why it came bundled with https://github.com/ggml-org/llama.cpp/tree/8f419181d1c20d8195148680df15b6f093cb1512/examples/speculative-simple

---

👤 **g2mt** commented on **2025-07-25** at **07:09:50**

I'm now able to load the draft model. It seems that the kv-cache tensor names were reused for both models. Prefixing them with the model name fixes it.

---

👤 **saood06** commented on **2025-07-25** at **07:47:27**

>I'm now able to load the draft model. It seems that the kv-cache tensor names were reused for both models. Prefixing them with the model name fixes it.

Nice. Did you get any accepted tokens?

---

👤 **g2mt** commented on **2025-07-25** at **09:02:33**

I think I got it working. For some reason ik_llama's slot.id is offset by 1, which tripped me off a bit.

A simple test of repeating a string shows it working:

```
curl -s http://localhost:9001/v1/chat/completions \
          -H "Content-Type: application/json" \
          -H "Authorization: Bearer no-key" \
          -d '{"model": "test","messages": [{"role": "user","content": "Repeat the following sentence, as is: The quick brown fox jumped over the lazy dog."}]}'
{"choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"The quick brown fox jumped over the lazy dog."}}],"created":1753433480,"model":"test","object":"chat.completion","usage":{"completion_tokens":14,"prompt_tokens":26,"total_tokens":40},"id":"chatcmpl-QK3CBenhWiSBeeuIs6UGs2yXCV5YpqRO","__verbose":{"content":"The quick brown fox jumped over the lazy dog.","generated_text":"The quick brown fox jumped over the lazy dog.",
```

Server logs do show the speculative decoding results being accepted:

```
VERB [            update_slots] speculative decoding result | tid="140737350637888" timestamp=1753433480 id_slot=0 accepted=12 total=13 new_n_past=39
```

It looks like it's working, but I think more testing is needed. If someone else could post more test results that would be great. I'll open the PR up for review now.

---

👤 **saood06** commented on **2025-07-25** at **09:12:46**

>If someone else could post more test results that would be great. I'll open the PR up for review now.

I'll try to do some tests within a day.

---

👤 **ikawrakow** commented on **2025-07-25** at **09:21:28**

@saood06 I'll be not able to review before August 7, so I have assigned you as a reviewer.

Hopefully more people will test.

---

👤 **saood06** commented on **2025-07-25** at **09:47:41**

> @saood06 I'll be not able to review before August 7, so I have assigned you as a reviewer.

I'll review and test it.