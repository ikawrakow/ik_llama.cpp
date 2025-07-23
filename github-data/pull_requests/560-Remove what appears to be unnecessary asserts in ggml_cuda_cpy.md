### 🔀 [#560](https://github.com/ikawrakow/ik_llama.cpp/pull/560) - Remove what appears to be unnecessary asserts in ggml_cuda_cpy

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-26 |
| **Updated** | 2025-06-27 |

---

#### Description

Not sure why the assert were there as it seems the code should handle tensor sizes greater than `INT_MAX`.

The funny part is that the assert is triggered when copying the KQ mask! I was able to trigger it using batch/u-batch of 16k tokens with a context of 32k tokens. Which means I should resurrect PR #28 as it is kind of ridiculous to be copying over 2 GB of data from the CPU to the GPU that could be 16X smaller if one used 1 bit per mask entry instead of a `fp16` value (or even `fp32` if not using FA).

After removing the assert everything seems to work fine.

But please test!

---

#### 💬 Conversation

👤 **Nexesenex** commented the **2025-06-27** at **15:29:27**:<br>

I merged this on my Croco.
My short benching session ok.
On Wizard 8x22B, 55/57 tensors offloaded on 3 different GPUs, and NKVO activated, no problem of corrupted inference.
And no losses of performances either.
Same goes on Miqu 70b full offload on triple GPU.