### ✨ [#67](https://github.com/ikawrakow/ik_llama.cpp/issues/67) - Feature Request: Elliminate/reduce unnecessary copies 

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2024-09-28 |

---

#### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

PR #66 does it for Phi-3(.5)-mini, with a non-negligible performance gain on GPUs. Architectures that could potentially benefit from the same optimization are Falcon, DBRX, Starcoder, Bert, Bloom, MPT, Qwen, Phi-2, GPT-2, Codeshell, OpenLM, GPT-Neox, ChatGLM.

### Motivation

Improve performance

### Possible Implementation

See #66