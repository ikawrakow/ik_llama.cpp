### ✨ [#433](https://github.com/ikawrakow/ik_llama.cpp/issues/433) - Feature Request: CORS support

| **Author** | `KCS-Mack` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-05-18 |
| **Updated** | 2025-05-18 |

---

#### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

With the original llama.cpp they added a flag to enable cors:

https://github.com/ggml-org/llama.cpp/pull/5781/commits/1e6a2f12c6453d7b5158b37c8a789fd3934af044

However I don't see that added to ik_llama(great work by the way, love this project!)

Is there any plans to enable CORS in the future?

### Motivation

I use an application endpoint that requires CORS to interract, It works with llama-cp with the --public-domain flag.

### Possible Implementation

_No response_

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-05-18** at **15:39:10**:<br>

You could use any reverse proxy to add this yourself e.g. nginx, caddy server, etc.

Also someone created a wrapper/reverse-proxy like thing to support tool calling and other openai style endpoint stuff it seems: https://github.com/ikawrakow/ik_llama.cpp/discussions/403#discussioncomment-13098276