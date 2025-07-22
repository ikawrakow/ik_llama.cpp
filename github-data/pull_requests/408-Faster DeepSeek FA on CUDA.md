### 🔀 [#408](https://github.com/ikawrakow/ik_llama.cpp/pull/408) - Faster DeepSeek FA on CUDA

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-11 |
| **Updated** | 2025-05-12 |

---

#### Description

This is a port of [this PR](https://github.com/ggml-org/llama.cpp/pull/13435) in mainline `llama.cpp`.

The main difference to PR #386 is that now the FA kernel takes advantage of the fact that the V tensor contains the same data as the K tensor (it is a view on the K cache with an offset given by the RoPE embedding size). Hence, one can reduce the number of loads by reusing K tiles when processing `V*softmax(K*Q)`.

To take advantage of this new kernel I had to change the way the K cache is organized. In mainline `llama.cpp` the K cache stores `(RoPE, NoPE)` parts in that order, and the FA kernel assumes this arrangement. But in `ik_llama.cpp` prior to this PR the K cache was stored as `(NoPE, RoPE)`. As there are several places where the views into the K cache can go wrong when building the graph, the PR should be tested more thoroughly before merging. I have tested all possible combinations of `mla` and `fa` using DeepSeek-Lite and it appears to work correctly, but still.

The next graph shows a TG performance comparison between the main branch (black) and this PR (red). Model is DeepSeek-Lite quantized with `Q4_0`, GPU is RTX-4080. We see nice performance improvements, but also a more peculiar behavior as a function of `N_KV`, the number of tokens in the KV cache.

![z10a](https://github.com/user-attachments/assets/1a7dfe72-580d-4be7-8868-5b95cfbd1e4d)

When `mla = 2` or `mla = 3` this PR has no effect on PP, so the next graph compares PP speed between the main branch (black) and the PR (red) for `mla = 1`. For reference I have also included PP performance for `mla = 3` with blue symbols. In case I ave not shown a graph such as this one, it illustrates what one gives up in terms of PP performance by using a mainline `llama.cpp` MLA-enabled GGUF for DeepSeek models. The difference is ~25% for `N_KV = 0` and nearly a factor of 2 at 60k tokens. The PR improves `mla = 1` performance by a few percent.

![z10b](https://github.com/user-attachments/assets/4dbdeafb-75a7-472f-bffa-b83a7f2019b5)

Finally, being curious about the peculiar TG behavior as a function of `N_KV`, I ran `sweep-bench` with the [llama.cpp PR]( https://github.com/ggml-org/llama.cpp/pull/13435), and the next graph shows a TG performance comparison between this PR and the mainline PR. We see that the two curves align very closely, so the strange behavior is not due to me screwing up with the port. I wonder if @JohannesGaessler is aware.

 
![z10c](https://github.com/user-attachments/assets/f18c6fc7-7026-4355-820d-409be77e079d)

---

#### 💬 Conversation

👤 **JohannesGaessler** commented the **2025-05-11** at **14:05:44**:<br>

An RTX 4080 has 76 streaming multiprocessor, the CUDA code assigns KV slices to SMs in chunks of size 256. So every 76*256=19456 tokens the size of biggest workload across all of the SMs increases and there is a dip in performance. These so-called quantization effects are much more noticeable with compute than with I/O so they become more pronounced if the I/O of a kernel is optimized.

---

👤 **Panchovix** commented the **2025-05-11** at **18:28:44**:<br>

Just tested on DeepSeek V3 0324 Q2_K_XL and it seems to have improved my t/s TG by about 1-2% (I guess with offloading there isn't much difference), but tested a smaller models (DeepSeek2 16B) on a single GPU (5090) and got about 8-12% speedup, so pretty nice!

This is on top of https://github.com/ikawrakow/ik_llama.cpp/pull/405 PR.

Now I'm gonna try https://github.com/ikawrakow/ik_llama.cpp/pull/409 on top of that PR and this PR.