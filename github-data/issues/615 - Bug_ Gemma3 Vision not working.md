### 🐛 [#615](https://github.com/ikawrakow/ik_llama.cpp/issues/615) - Bug: Gemma3 Vision not working

| **Author** | `erazortt` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-15 |
| **Updated** | 2025-07-19 |

---

#### Description

### What happened?

Using the follollowing command the vision is not working:
llama-server.exe -m models/gemma-3-27b-it.gguf --mmproj models/gemma-3-27b-it-mmproj-bf16.gguf --temp 1 --top-k 64 --top-p 0.95 --min-p 0.01 -ngl 63 -c 32768 -ctk q8_0 -ctv q8_0 --flash-attn --no-kv-offload --port 10000

When using exactly the same command line on llama.cpp it works.

### Name and Version

$ ./llama-server.exe --version
version: 1 (8c2a6ee)
built with MSVC 19.44.35211.0 for


### What operating system are you seeing the problem on?

Windows

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **jmcook** commented the **2025-07-16** at **03:36:05**:<br>

That's funny, I was just trying the same thing tonight and noticed the same thing!

---

👤 **ikawrakow** commented the **2025-07-16** at **09:18:39**:<br>

Sorry, there is no vision support in `ik_llama.cpp` at all. As I know nothing about vision or multi-modality, my suggestion is to try to convince @ngxson to contribute the multi-modality library he created for `llama.cpp` also to `ik_llama.cpp`.

---

👤 **ikawrakow** commented the **2025-07-19** at **09:27:13**:<br>

I think I'll close this one. A feature request can be opened instead.