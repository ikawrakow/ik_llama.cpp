### 🐛 [#364](https://github.com/ikawrakow/ik_llama.cpp/pull/364) - Fix FA bug on AVX2

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-01 |
| **Updated** | 2025-05-02 |

---

#### Description

The bug was quite subtle: we have `Q8_0` K-cache, so we need to quantize the `Q` tensor to the appropriate quantization type (`vec_dot_type` in `ggml` lingo) that differs from platform to platform. We pick correctly the type. But then we notice that it is a GQA case, so we repack  the K tensor to `Q8_0_R8` for faster processing, but still use the `vec_dot_type` selected based on `K` being `Q8_0`. On `Zen4` and `ARM_NEON` the `vet_dot_type` is the same, so everything works fine. But on `AVX2` the `vec_dot_type` changes, and we get gibberish (or even an assert for a NaN value). 

The bug was introduced in my recent CPU FA optimization round (#351) 

Closes #363

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-02** at **05:09:05**:<br>

It looks like this does not fully fix #363, but I'll merge it to not have 2 real bugs stay on the main branch.