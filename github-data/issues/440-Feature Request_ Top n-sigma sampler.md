### ✨ [#440](https://github.com/ikawrakow/ik_llama.cpp/issues/440) - Feature Request: Top n-sigma sampler

| **Author** | `Ph0rk0z` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-20 |
| **Updated** | 2025-06-03 |

---

#### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

It's another good sampler like XTC and DRY. Was just added recently: https://github.com/ggml-org/llama.cpp/pull/13264

I've not checked to see how different sampling is here from mainline and if it's possible to just copy the PR or if that is a nono.

### Motivation

I see people using/recommending it and do not have it :P

Seems like relatively low hanging fruit on the surface, unlike, say vision in the server. (where we don't have a good large MoE with vision; llama doesn't count)

### Possible Implementation

_No response_

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-20** at **15:47:10**:<br>

So, the quoted PR just integrates it into the standard `llama.cpp` sampling mechanism. The actual sampler is implemented in their PR 11233. I looked at 11233, and it is a pretty trivial thing, so very easy to implement. I had never actually looked at the sampling code, but a quick check shows that it is not a copy/paste. Also this has been completely reorganized in mainline (they just love pushing pieces of code from here to there). Here sampling is part of `common`, over there it is now part of `llama.cpp` itself. So, adding a new sampler involves me first getting familiar with how sampling is done in this fork.

---

👤 **Ph0rk0z** commented the **2025-06-03** at **13:58:36**:<br>

https://github.com/ikawrakow/ik_llama.cpp/pull/489