### ✨ [#407](https://github.com/ikawrakow/ik_llama.cpp/issues/407) - Feature Request: Support for function calling in llama-server

| **Author** | `vijaysaayi` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-05-11 |
| **Updated** | 2025-06-08 |

---

#### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Could you add support for function calling supported in llama.cpp ?
- https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md

Currently it is not supported.
- https://github.com/ikawrakow/ik_llama.cpp/blob/36e6e888b75ae93fb5aac212bb0e147d8379ae23/examples/server/utils.hpp#L394


### Motivation

Tool calling will enable agent scenarios.

### Possible Implementation

https://github.com/ggml-org/llama.cpp/pull/9639

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-12** at **05:38:22**:<br>

I have never used function calling myself, so I'm not familiar with this feature.

Help will be appreciated.

---

👤 **vijaysaayi** commented the **2025-05-16** at **15:47:05**:<br>

Thanks for all the efforts on this. Would it be possible to update to latest llama.cpp (These functionalities are implemented)

---

👤 **ikawrakow** commented the **2025-05-16** at **15:52:09**:<br>

The code here has not been synced with `llama.cpp` since last August, and as a result the two code bases have totally diverged. Almost nothing is just a copy/paste from upstream.

---

👤 **ubergarm** commented the **2025-05-18** at **15:38:27**:<br>

@vijaysaayi Check out this wrapper/reverse-proxy which might be able to do what you want: https://github.com/ikawrakow/ik_llama.cpp/discussions/403#discussioncomment-13098276

---

👤 **vijaysaayi** commented the **2025-05-26** at **07:57:13**:<br>

Thanks for sharing this. I will check this out.

---

👤 **mtcl** commented the **2025-06-08** at **06:07:47**:<br>

@vijaysaayi let me know if you need any help with the function calling wrapper. Here is the video walkthrough of it.  https://www.youtube.com/watch?v=JGo9HfkzAmc