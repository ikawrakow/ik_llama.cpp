### ✨ [#228](https://github.com/ikawrakow/ik_llama.cpp/issues/228) - Feature Request: create tool to offline repack models

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-23 |
| **Updated** | 2025-03-21 |

---

#### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description


Add a tool to repack an existing quantized model to `_R4/_R8` quants and store the result on disk for later use.


### Motivation

Run time repacking increases performance, but can significantly prolong model loading for very large models such as DeepSeekV3/R1. One can of course re-quantize the model to `_R4/_R8` quants, but the original `f16/bf16` model may not be available (because, e.g., it is extremely large and the user did not download). Hence, it would be useful to have a tool to repack an existing quantized model to `_R4/_R8` quants and store the resulting model on disk.    

### Possible Implementation

_No response_